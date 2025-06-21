// CrewAI Workflow Manager Frontend JavaScript

class WorkflowManager {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.workflowRunning = false;
        this.currentWorkflowId = null;
        this.uploadedDocuments = [];
        this.websocket = null;
        
        this.initializeEventListeners();
        this.checkServerStatus();
        this.loadSavedConfiguration();
    }

    initializeEventListeners() {
        // Configuration events
        document.getElementById('uploadBtn').addEventListener('click', () => this.uploadDocuments());
        document.getElementById('documentUpload').addEventListener('change', (e) => this.handleFileSelection(e));
        
        // Workflow control events
        document.getElementById('startWorkflowBtn').addEventListener('click', () => this.startWorkflow());
        document.getElementById('stopWorkflowBtn').addEventListener('click', () => this.stopWorkflow());
        document.getElementById('testModelsBtn').addEventListener('click', () => this.testModels());
        document.getElementById('generateDataBtn').addEventListener('click', () => this.generateDataOnly());
        document.getElementById('implementRagBtn').addEventListener('click', () => this.implementRagOnly());
        
        // Utility events
        document.getElementById('clearLogsBtn').addEventListener('click', () => this.clearLogs());
        
        // Save configuration on change
        ['dataGenModel', 'embeddingModel', 'rerankingModel', 'openaiKey', 'ollamaUrl', 'enableGpuOptimization'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.saveConfiguration());
            }
        });

        // Drag and drop for file upload
        this.setupDragAndDrop();
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('documentUpload').parentElement;
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });

        uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        document.getElementById('documentUpload').files = files;
        this.handleFileSelection({ target: { files } });
    }

    async checkServerStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (response.ok) {
                this.updateServerStatus('online');
                this.connectWebSocket();
            } else {
                this.updateServerStatus('offline');
            }
        } catch (error) {
            this.updateServerStatus('offline');
            this.logMessage('Server is not running. Please start the backend server.', 'error');
        }
    }

    updateServerStatus(status) {
        // Add status indicator to navbar if it doesn't exist
        let statusIndicator = document.querySelector('.server-status');
        if (!statusIndicator) {
            statusIndicator = document.createElement('span');
            statusIndicator.className = 'server-status ms-3';
            document.querySelector('.navbar-brand').appendChild(statusIndicator);
        }
        
        statusIndicator.innerHTML = `
            <span class="status-indicator ${status}"></span>
            Server ${status === 'online' ? 'Online' : 'Offline'}
        `;
    }

    connectWebSocket() {
        try {
            this.websocket = new WebSocket('ws://localhost:8000/ws');
            
            this.websocket.onopen = () => {
                this.logMessage('Connected to workflow updates', 'success');
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onclose = () => {
                this.logMessage('Disconnected from workflow updates', 'warning');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
            
            this.websocket.onerror = (error) => {
                this.logMessage('WebSocket error: ' + error.message, 'error');
            };
        } catch (error) {
            this.logMessage('Failed to connect WebSocket: ' + error.message, 'error');
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'workflow_progress':
                this.updateWorkflowProgress(data.step, data.status, data.progress);
                break;
            case 'log':
                this.logMessage(data.message, data.level);
                break;
            case 'workflow_complete':
                this.handleWorkflowComplete(data);
                break;
            case 'workflow_error':
                this.handleWorkflowError(data);
                break;
        }
    }

    handleFileSelection(event) {
        const files = Array.from(event.target.files);
        const fileList = document.getElementById('documentList');
        
        if (files.length === 0) return;
        
        // Clear the "no documents" message
        fileList.innerHTML = '';
        
        files.forEach(file => {
            const fileItem = this.createFileItem(file);
            fileList.appendChild(fileItem);
        });
    }

    createFileItem(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'document-item fade-in';
        
        const fileIcon = this.getFileIcon(file.name);
        const fileSize = this.formatFileSize(file.size);
        
        fileItem.innerHTML = `
            <div class="document-info">
                <i class="${fileIcon} document-icon"></i>
                <div>
                    <div class="document-name">${file.name}</div>
                    <div class="document-size">${fileSize}</div>
                </div>
            </div>
            <div class="document-actions">
                <button class="btn btn-sm btn-outline-danger" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        return fileItem;
    }

    getFileIcon(filename) {
        const extension = filename.split('.').pop().toLowerCase();
        switch (extension) {
            case 'pdf': return 'fas fa-file-pdf';
            case 'csv': return 'fas fa-file-csv';
            case 'txt': return 'fas fa-file-alt';
            default: return 'fas fa-file';
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async uploadDocuments() {
        const fileInput = document.getElementById('documentUpload');
        const files = fileInput.files;
        
        if (files.length === 0) {
            this.showAlert('Please select files to upload', 'warning');
            return;
        }

        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.classList.add('loading');
        uploadBtn.disabled = true;

        try {
            const formData = new FormData();
            Array.from(files).forEach(file => {
                formData.append('files', file);
            });

            const response = await fetch(`${this.apiBaseUrl}/upload-documents`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                this.uploadedDocuments = result.documents;
                this.logMessage(`Successfully uploaded ${files.length} document(s)`, 'success');
                this.showAlert('Documents uploaded successfully!', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }
        } catch (error) {
            this.logMessage(`Upload failed: ${error.message}`, 'error');
            this.showAlert(`Upload failed: ${error.message}`, 'error');
        } finally {
            uploadBtn.classList.remove('loading');
            uploadBtn.disabled = false;
        }
    }

    async startWorkflow() {
        if (!this.validateConfiguration()) {
            return;
        }

        const config = this.getConfiguration();
        
        try {
            this.setWorkflowRunning(true);
            this.resetWorkflowProgress();
            
            const response = await fetch(`${this.apiBaseUrl}/start-workflow`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                const result = await response.json();
                this.currentWorkflowId = result.workflow_id;
                this.logMessage('Workflow started successfully', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start workflow');
            }
        } catch (error) {
            this.logMessage(`Failed to start workflow: ${error.message}`, 'error');
            this.setWorkflowRunning(false);
            this.showAlert(`Failed to start workflow: ${error.message}`, 'error');
        }
    }

    async stopWorkflow() {
        if (!this.currentWorkflowId) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/stop-workflow/${this.currentWorkflowId}`, {
                method: 'POST'
            });

            if (response.ok) {
                this.logMessage('Workflow stopped', 'warning');
                this.setWorkflowRunning(false);
            } else {
                throw new Error('Failed to stop workflow');
            }
        } catch (error) {
            this.logMessage(`Failed to stop workflow: ${error.message}`, 'error');
        }
    }

    async testModels() {
        const config = this.getConfiguration();
        
        try {
            const testBtn = document.getElementById('testModelsBtn');
            testBtn.classList.add('loading');
            testBtn.disabled = true;

            const response = await fetch(`${this.apiBaseUrl}/test-models`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                const result = await response.json();
                this.displayTestResults(result);
                this.logMessage('Model testing completed', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Model testing failed');
            }
        } catch (error) {
            this.logMessage(`Model testing failed: ${error.message}`, 'error');
            this.showAlert(`Model testing failed: ${error.message}`, 'error');
        } finally {
            const testBtn = document.getElementById('testModelsBtn');
            testBtn.classList.remove('loading');
            testBtn.disabled = false;
        }
    }

    async generateDataOnly() {
        if (!this.validateConfiguration()) {
            return;
        }

        const config = this.getConfiguration();
        config.workflow_type = 'data_generation_only';
        
        try {
            this.setWorkflowRunning(true);
            this.resetWorkflowProgress();
            
            const response = await fetch(`${this.apiBaseUrl}/start-workflow`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                const result = await response.json();
                this.currentWorkflowId = result.workflow_id;
                this.logMessage('Data generation workflow started', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start data generation');
            }
        } catch (error) {
            this.logMessage(`Failed to start data generation: ${error.message}`, 'error');
            this.setWorkflowRunning(false);
        }
    }

    async implementRagOnly() {
        if (!this.validateConfiguration()) {
            return;
        }

        const config = this.getConfiguration();
        config.workflow_type = 'rag_only';
        
        try {
            this.setWorkflowRunning(true);
            this.resetWorkflowProgress();
            
            const response = await fetch(`${this.apiBaseUrl}/start-workflow`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                const result = await response.json();
                this.currentWorkflowId = result.workflow_id;
                this.logMessage('RAG implementation workflow started', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start RAG implementation');
            }
        } catch (error) {
            this.logMessage(`Failed to start RAG implementation: ${error.message}`, 'error');
            this.setWorkflowRunning(false);
        }
    }

    validateConfiguration() {
        const config = this.getConfiguration();
        
        if (!config.data_generation_model) {
            this.showAlert('Please select a data generation model', 'warning');
            return false;
        }
        
        if (!config.embedding_model) {
            this.showAlert('Please select an embedding model', 'warning');
            return false;
        }
        
        if (this.uploadedDocuments.length === 0) {
            this.showAlert('Please upload at least one document', 'warning');
            return false;
        }
        
        // Check if OpenAI models are selected but no API key provided
        if ((config.data_generation_model.startsWith('openai:') || 
             config.embedding_model.startsWith('openai:')) && 
            !config.openai_api_key) {
            this.showAlert('OpenAI API key is required for OpenAI models', 'warning');
            return false;
        }
        
        return true;
    }

    getConfiguration() {
        return {
            data_generation_model: document.getElementById('dataGenModel').value,
            embedding_model: document.getElementById('embeddingModel').value,
            reranking_model: document.getElementById('rerankingModel').value,
            openai_api_key: document.getElementById('openaiKey').value,
            ollama_url: document.getElementById('ollamaUrl').value,
            enable_gpu_optimization: document.getElementById('enableGpuOptimization').checked,
            documents: this.uploadedDocuments
        };
    }

    saveConfiguration() {
        const config = this.getConfiguration();
        localStorage.setItem('crewai_workflow_config', JSON.stringify(config));
    }

    loadSavedConfiguration() {
        const saved = localStorage.getItem('crewai_workflow_config');
        if (saved) {
            try {
                const config = JSON.parse(saved);
                
                if (config.data_generation_model) {
                    document.getElementById('dataGenModel').value = config.data_generation_model;
                }
                if (config.embedding_model) {
                    document.getElementById('embeddingModel').value = config.embedding_model;
                }
                if (config.reranking_model) {
                    document.getElementById('rerankingModel').value = config.reranking_model;
                }
                if (config.openai_api_key) {
                    document.getElementById('openaiKey').value = config.openai_api_key;
                }
                if (config.ollama_url) {
                    document.getElementById('ollamaUrl').value = config.ollama_url;
                }
                if (config.enable_gpu_optimization !== undefined) {
                    document.getElementById('enableGpuOptimization').checked = config.enable_gpu_optimization;
                }
            } catch (error) {
                console.error('Failed to load saved configuration:', error);
            }
        }
    }

    setWorkflowRunning(running) {
        this.workflowRunning = running;
        
        document.getElementById('startWorkflowBtn').disabled = running;
        document.getElementById('stopWorkflowBtn').disabled = !running;
        document.getElementById('testModelsBtn').disabled = running;
        document.getElementById('generateDataBtn').disabled = running;
        document.getElementById('implementRagBtn').disabled = running;
        
        if (!running) {
            this.currentWorkflowId = null;
        }
    }

    resetWorkflowProgress() {
        const steps = ['document-processing', 'model-selection', 'data-generation', 'rag-implementation', 'optimization'];
        
        steps.forEach(step => {
            const stepElement = document.getElementById(`step-${step}`);
            if (stepElement) {
                stepElement.className = 'step-item';
                stepElement.querySelector('.step-icon').className = 'fas fa-circle-notch step-icon';
                stepElement.querySelector('.step-status').textContent = 'Pending';
            }
        });
        
        document.getElementById('overallProgress').style.width = '0%';
        document.getElementById('overallProgress').textContent = '0%';
    }

    updateWorkflowProgress(step, status, progress) {
        const stepElement = document.getElementById(`step-${step}`);
        if (stepElement) {
            stepElement.className = `step-item ${status}`;
            
            const icon = stepElement.querySelector('.step-icon');
            const statusText = stepElement.querySelector('.step-status');
            
            switch (status) {
                case 'active':
                    icon.className = 'fas fa-spinner step-icon';
                    statusText.textContent = 'In Progress';
                    break;
                case 'completed':
                    icon.className = 'fas fa-check-circle step-icon';
                    statusText.textContent = 'Completed';
                    break;
                case 'error':
                    icon.className = 'fas fa-exclamation-circle step-icon';
                    statusText.textContent = 'Error';
                    break;
            }
        }
        
        if (progress !== undefined) {
            document.getElementById('overallProgress').style.width = `${progress}%`;
            document.getElementById('overallProgress').textContent = `${progress}%`;
        }
    }

    handleWorkflowComplete(data) {
        this.setWorkflowRunning(false);
        this.logMessage('Workflow completed successfully!', 'success');
        this.displayResults(data.results);
        this.showAlert('Workflow completed successfully!', 'success');
    }

    handleWorkflowError(data) {
        this.setWorkflowRunning(false);
        this.logMessage(`Workflow failed: ${data.error}`, 'error');
        this.showAlert(`Workflow failed: ${data.error}`, 'error');
    }

    displayTestResults(results) {
        const modal = new bootstrap.Modal(document.getElementById('statusModal'));
        const modalContent = document.getElementById('modalContent');
        
        let html = '<h6>Model Test Results</h6>';
        
        Object.entries(results).forEach(([model, result]) => {
            const status = result.success ? 'success' : 'danger';
            const icon = result.success ? 'check-circle' : 'exclamation-circle';
            
            html += `
                <div class="alert alert-${status} d-flex align-items-center">
                    <i class="fas fa-${icon} me-2"></i>
                    <div>
                        <strong>${model}</strong><br>
                        ${result.message}
                        ${result.response_time ? `<br><small>Response time: ${result.response_time}ms</small>` : ''}
                    </div>
                </div>
            `;
        });
        
        modalContent.innerHTML = html;
        modal.show();
    }

    displayResults(results) {
        const resultsPanel = document.getElementById('resultsPanel');
        resultsPanel.innerHTML = '';
        
        if (!results || results.length === 0) {
            resultsPanel.innerHTML = '<div class="text-muted text-center py-3">No results available</div>';
            return;
        }
        
        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item fade-in';
            
            resultItem.innerHTML = `
                <div class="result-info">
                    <h6>${result.title}</h6>
                    <small>${result.description}</small>
                </div>
                <div class="result-actions">
                    <button class="btn btn-sm btn-primary" onclick="workflowManager.downloadResult('${result.id}')">
                        <i class="fas fa-download me-1"></i>Download
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="workflowManager.viewResult('${result.id}')">
                        <i class="fas fa-eye me-1"></i>View
                    </button>
                </div>
            `;
            
            resultsPanel.appendChild(resultItem);
        });
    }

    async downloadResult(resultId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/download-result/${resultId}`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `result_${resultId}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            this.showAlert(`Download failed: ${error.message}`, 'error');
        }
    }

    async viewResult(resultId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/view-result/${resultId}`);
            if (response.ok) {
                const result = await response.json();
                
                const modal = new bootstrap.Modal(document.getElementById('statusModal'));
                const modalContent = document.getElementById('modalContent');
                
                modalContent.innerHTML = `
                    <h6>${result.title}</h6>
                    <pre class="bg-light p-3 rounded" style="max-height: 400px; overflow-y: auto;">${JSON.stringify(result.data, null, 2)}</pre>
                `;
                
                modal.show();
            } else {
                throw new Error('Failed to load result');
            }
        } catch (error) {
            this.showAlert(`Failed to view result: ${error.message}`, 'error');
        }
    }

    logMessage(message, level = 'info') {
        const outputPanel = document.getElementById('outputPanel');
        const timestamp = new Date().toLocaleTimeString();
        
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry fade-in';
        
        logEntry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-level-${level}">[${level.toUpperCase()}]</span>
            ${message}
        `;
        
        outputPanel.appendChild(logEntry);
        outputPanel.scrollTop = outputPanel.scrollHeight;
        
        // Keep only last 100 log entries
        while (outputPanel.children.length > 100) {
            outputPanel.removeChild(outputPanel.firstChild);
        }
    }

    clearLogs() {
        const outputPanel = document.getElementById('outputPanel');
        outputPanel.innerHTML = '<div class="text-muted">Logs cleared...</div>';
    }

    showAlert(message, type = 'info') {
        // Create toast notification
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast element after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '1055';
        document.body.appendChild(container);
        return container;
    }
}

// Initialize the workflow manager when the page loads
let workflowManager;
document.addEventListener('DOMContentLoaded', () => {
    workflowManager = new WorkflowManager();
});
