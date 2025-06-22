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
        
        // Model refresh events
        document.getElementById('refreshModelsBtn').addEventListener('click', () => this.refreshOllamaModels());
        
        // Save configuration on change
        ['dataGenModel', 'embeddingModel', 'rerankingModel', 'openaiKey', 'ollamaUrl', 'enableGpuOptimization'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => {
                    this.saveConfiguration();
                    // Refresh models when Ollama URL changes
                    if (id === 'ollamaUrl') {
                        this.loadAvailableModels();
                    }
                });
            }
        });

        // Drag and drop for file upload
        this.setupDragAndDrop();
        
        // Load available models on startup
        this.loadAvailableModels();
        
        // Initialize tooltips
        this.initializeTooltips();
        
        // Load existing results
        this.loadResultsFromBackend();
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
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
            // Load results from backend
            this.loadResultsFromBackend();
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

    async loadResultsFromBackend() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/list-results`);
            if (response.ok) {
                const data = await response.json();
                this.displayBackendResults(data.results);
            } else {
                const resultsPanel = document.getElementById('resultsPanel');
                resultsPanel.innerHTML = '<div class="text-muted text-center py-3">No results available</div>';
            }
        } catch (error) {
            console.error('Failed to load results:', error);
            const resultsPanel = document.getElementById('resultsPanel');
            resultsPanel.innerHTML = '<div class="text-muted text-center py-3">Failed to load results</div>';
        }
    }

    displayBackendResults(results) {
        const resultsPanel = document.getElementById('resultsPanel');
        resultsPanel.innerHTML = '';
        
        if (!results || results.length === 0) {
            resultsPanel.innerHTML = '<div class="text-muted text-center py-3">No results available</div>';
            return;
        }
        
        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item fade-in';
            
            // Enhanced display for different result types
            let typeIcon = 'fas fa-file';
            let typeColor = 'secondary';
            let additionalInfo = '';
            
            if (result.type === 'alpaca_dataset') {
                typeIcon = 'fas fa-database';
                typeColor = 'success';
                additionalInfo = '<span class="badge bg-success ms-2">Alpaca Dataset</span>';
            } else if (result.type === 'rag_implementation') {
                typeIcon = 'fas fa-search';
                typeColor = 'info';
                additionalInfo = '<span class="badge bg-info ms-2">RAG System</span>';
            } else if (result.type === 'synthetic_data') {
                typeIcon = 'fas fa-magic';
                typeColor = 'warning';
                additionalInfo = '<span class="badge bg-warning ms-2">Synthetic Data</span>';
            }
            
            resultItem.innerHTML = `
                <div class="result-info">
                    <div class="d-flex align-items-center">
                        <i class="${typeIcon} text-${typeColor} me-2"></i>
                        <div>
                            <h6 class="mb-1">${result.title}${additionalInfo}</h6>
                            <small class="text-muted">${result.description}</small>
                            ${result.created_at ? `<br><small class="text-muted">Created: ${new Date(result.created_at).toLocaleString()}</small>` : ''}
                        </div>
                    </div>
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

    async loadAvailableModels() {
        try {
            this.logMessage('Loading available models...', 'info');
            
            // Get current Ollama URL from configuration
            const ollamaUrl = document.getElementById('ollamaUrl').value || 'http://host.docker.internal:11434';
            
            // Get Ollama models
            const response = await fetch(`${this.apiBaseUrl}/ollama-models?ollama_url=${encodeURIComponent(ollamaUrl)}`);
            if (response.ok) {
                const data = await response.json();
                if (data.error) {
                    this.logMessage(`Ollama connection failed (${ollamaUrl}): ${data.error}`, 'warning');
                    this.logMessage('Only OpenAI models will be available. Check Ollama server status.', 'warning');
                    this.populateModelDropdowns([]); // Still populate with OpenAI options
                } else {
                    this.populateModelDropdowns(data.models);
                    this.logMessage(`Successfully loaded ${data.models.length} Ollama models from ${ollamaUrl}`, 'success');
                }
            } else {
                const errorText = await response.text();
                this.logMessage(`Failed to connect to Ollama server (${ollamaUrl}): ${response.status} - ${errorText}`, 'warning');
                this.logMessage('Only OpenAI models will be available. Verify Ollama server is running.', 'warning');
                this.populateModelDropdowns([]); // Still populate with OpenAI options
            }
        } catch (error) {
            this.logMessage(`Error loading models from ${ollamaUrl}: ${error.message}`, 'error');
            this.logMessage('Only OpenAI models will be available. Check network connectivity.', 'warning');
            this.populateModelDropdowns([]); // Still populate with OpenAI options
        }
    }

    categorizeModel(modelName) {
        const name = modelName.toLowerCase();
        
        // Dedicated embedding models
        const embeddingKeywords = ['embed', 'nomic', 'bge', 'e5', 'sentence', 'all-minilm'];
        const isEmbedding = embeddingKeywords.some(keyword => name.includes(keyword));
        
        // Small models good for embeddings but can also do text generation
        const smallModels = ['phi', 'gemma:2b', 'qwen:0.5b', 'qwen:1.8b'];
        const isSmall = smallModels.some(model => name.includes(model));
        
        // Large models better for data generation
        const largeModels = ['llama', 'mistral', 'codellama', 'qwen:7b', 'qwen:14b', 'gemma:7b'];
        const isLarge = largeModels.some(model => name.includes(model));
        
        return {
            isEmbedding,
            isSmall,
            isLarge,
            canGenerate: !isEmbedding, // Most models can generate text except pure embedding models
            canEmbed: isEmbedding || isSmall, // Embedding models + small models work for embeddings
            recommended: this.getModelRecommendation(name)
        };
    }
    
    getModelRecommendation(modelName) {
        const name = modelName.toLowerCase();
        
        // Embedding model recommendations
        if (name.includes('nomic-embed')) return 'Excellent for embeddings';
        if (name.includes('all-minilm')) return 'Good general-purpose embeddings';
        if (name.includes('bge')) return 'High-quality embeddings';
        
        // Data generation recommendations
        if (name.includes('llama') && (name.includes('7b') || name.includes('8b'))) return 'Excellent for data generation';
        if (name.includes('mistral')) return 'Great for creative text generation';
        if (name.includes('codellama')) return 'Best for code generation';
        if (name.includes('qwen') && name.includes('7b')) return 'Good multilingual model';
        if (name.includes('gemma')) return 'Efficient and capable';
        if (name.includes('phi')) return 'Fast and lightweight';
        
        return '';
    }

    populateModelDropdowns(ollamaModels) {
        const dataGenSelect = document.getElementById('dataGenModel');
        const embeddingSelect = document.getElementById('embeddingModel');
        const rerankingSelect = document.getElementById('rerankingModel');

        // Clear existing options and set placeholder
        [dataGenSelect, embeddingSelect, rerankingSelect].forEach(select => {
            select.innerHTML = '<option value="">Select Model...</option>';
        });

        // Categorize Ollama models
        const categorizedModels = ollamaModels.map(model => ({
            name: model,
            ...this.categorizeModel(model)
        }));

        // Add data generation models (prioritize larger models)
        const dataGenModels = categorizedModels
            .filter(model => model.canGenerate)
            .sort((a, b) => {
                // Prioritize: large models > small models > others
                if (a.isLarge && !b.isLarge) return -1;
                if (!a.isLarge && b.isLarge) return 1;
                if (a.isSmall && !b.isSmall) return -1;
                if (!a.isSmall && b.isSmall) return 1;
                return a.name.localeCompare(b.name);
            });

        dataGenModels.forEach(model => {
            const option = document.createElement('option');
            option.value = `ollama:${model.name}`;
            const recommendation = model.recommended ? ` (${model.recommended})` : '';
            option.textContent = `Ollama - ${model.name}${recommendation}`;
            if (model.isLarge) option.style.fontWeight = 'bold';
            dataGenSelect.appendChild(option);
        });

        // Add embedding models (prioritize dedicated embedding models)
        const embeddingModels = categorizedModels
            .filter(model => model.canEmbed)
            .sort((a, b) => {
                // Prioritize: embedding models > small models > others
                if (a.isEmbedding && !b.isEmbedding) return -1;
                if (!a.isEmbedding && b.isEmbedding) return 1;
                if (a.isSmall && !b.isSmall) return -1;
                if (!a.isSmall && b.isSmall) return 1;
                return a.name.localeCompare(b.name);
            });

        embeddingModels.forEach(model => {
            const option = document.createElement('option');
            option.value = `ollama:${model.name}`;
            const recommendation = model.recommended ? ` (${model.recommended})` : '';
            option.textContent = `Ollama - ${model.name}${recommendation}`;
            if (model.isEmbedding) option.style.fontWeight = 'bold';
            embeddingSelect.appendChild(option);
        });

        // For reranking, use embedding models (since Ollama doesn't have dedicated rerankers)
        embeddingModels.forEach(model => {
            const option = document.createElement('option');
            option.value = `ollama:${model.name}`;
            const recommendation = model.recommended ? ` (${model.recommended})` : '';
            option.textContent = `Ollama - ${model.name}${recommendation}`;
            if (model.isEmbedding) option.style.fontWeight = 'bold';
            rerankingSelect.appendChild(option);
        });

        // Add OpenAI options
        const openaiDataGenModels = [
            { value: 'openai:gpt-4', text: 'OpenAI - GPT-4 (Premium quality)' },
            { value: 'openai:gpt-4-turbo', text: 'OpenAI - GPT-4 Turbo (Fast & capable)' },
            { value: 'openai:gpt-3.5-turbo', text: 'OpenAI - GPT-3.5 Turbo (Cost effective)' }
        ];

        const openaiEmbeddingModels = [
            { value: 'openai:text-embedding-3-large', text: 'OpenAI - Embedding 3 Large (Best quality)' },
            { value: 'openai:text-embedding-3-small', text: 'OpenAI - Embedding 3 Small (Good balance)' },
            { value: 'openai:text-embedding-ada-002', text: 'OpenAI - Ada 002 (Legacy)' }
        ];

        // Add separators and OpenAI options
        if (dataGenModels.length > 0) {
            const separator = document.createElement('option');
            separator.disabled = true;
            separator.textContent = '── OpenAI Models ──';
            dataGenSelect.appendChild(separator);
        }

        openaiDataGenModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.text;
            dataGenSelect.appendChild(option);
        });

        if (embeddingModels.length > 0) {
            const separator = document.createElement('option');
            separator.disabled = true;
            separator.textContent = '── OpenAI Models ──';
            embeddingSelect.appendChild(separator);
            
            const separator2 = document.createElement('option');
            separator2.disabled = true;
            separator2.textContent = '── OpenAI Models ──';
            rerankingSelect.appendChild(separator2);
        }

        openaiEmbeddingModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.text;
            embeddingSelect.appendChild(option);
            
            const option2 = document.createElement('option');
            option2.value = model.value;
            option2.textContent = model.text;
            rerankingSelect.appendChild(option2);
        });

        // Add auto-select recommendations if models are available
        this.addAutoSelectOptions(categorizedModels);
    }
    
    addAutoSelectOptions(categorizedModels) {
        // Find best recommendations
        const bestDataGen = categorizedModels.find(m => m.isLarge && m.name.includes('llama')) ||
                           categorizedModels.find(m => m.isLarge) ||
                           categorizedModels.find(m => m.canGenerate);
                           
        const bestEmbedding = categorizedModels.find(m => m.isEmbedding && m.name.includes('nomic')) ||
                             categorizedModels.find(m => m.isEmbedding) ||
                             categorizedModels.find(m => m.canEmbed);

        if (bestDataGen || bestEmbedding) {
            // Add auto-select button
            const autoSelectBtn = document.createElement('button');
            autoSelectBtn.className = 'btn btn-sm btn-outline-success mt-2';
            autoSelectBtn.innerHTML = '<i class="fas fa-magic me-1"></i>Auto-select Recommended';
            autoSelectBtn.onclick = () => this.autoSelectModels(bestDataGen, bestEmbedding);
            
            const refreshBtn = document.getElementById('refreshModelsBtn');
            refreshBtn.parentNode.appendChild(autoSelectBtn);
        }
    }
    
    autoSelectModels(bestDataGen, bestEmbedding) {
        if (bestDataGen) {
            document.getElementById('dataGenModel').value = `ollama:${bestDataGen.name}`;
        }
        if (bestEmbedding) {
            document.getElementById('embeddingModel').value = `ollama:${bestEmbedding.name}`;
            document.getElementById('rerankingModel').value = `ollama:${bestEmbedding.name}`;
        }
        
        this.saveConfiguration();
        this.showAlert('Recommended models selected!', 'success');
    }

    async refreshOllamaModels() {
        const refreshBtn = document.getElementById('refreshModelsBtn');
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Refreshing...';
        }

        await this.loadAvailableModels();

        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.innerHTML = '<i class="fas fa-sync me-2"></i>Refresh Models';
        }
    }
}

// Initialize the workflow manager when the page loads
let workflowManager;
document.addEventListener('DOMContentLoaded', () => {
    workflowManager = new WorkflowManager();
});

// Troubleshooting Manager Class
class TroubleshootingManager {
    constructor(workflowManager) {
        this.workflowManager = workflowManager;
        this.currentTestResults = {};
        this.troubleshootLogs = [];
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Test buttons
        document.getElementById('runApiHealthTest').addEventListener('click', () => this.runApiHealthTest());
        document.getElementById('runDockerOllamaTest').addEventListener('click', () => this.runDockerOllamaTest());
        document.getElementById('runModelDebugTest').addEventListener('click', () => this.runModelDebugTest());
        document.getElementById('runWorkflowModelTest').addEventListener('click', () => this.runWorkflowModelTest());
        document.getElementById('runLlmDebugTest').addEventListener('click', () => this.runLlmDebugTest());
        document.getElementById('runCrewWorkflowTest').addEventListener('click', () => this.runCrewWorkflowTest());
        document.getElementById('runOllamaWorkflowTest').addEventListener('click', () => this.runOllamaWorkflowTest());
        document.getElementById('runAllTests').addEventListener('click', () => this.runAllTests());
        
        // Utility buttons
        document.getElementById('clearTroubleshootLogs').addEventListener('click', () => this.clearLogs());
        document.getElementById('exportTroubleshootResults').addEventListener('click', () => this.exportResults());
        
        // Debug model refresh button
        document.getElementById('refreshDebugModelsBtn').addEventListener('click', () => this.loadDebugModels());
        
        // Wiki button
        document.getElementById('openTroubleshootingWiki').addEventListener('click', () => this.openTroubleshootingWiki());
        
        // View toggle
        document.getElementById('outputViewLogs').addEventListener('change', () => this.toggleView('logs'));
        document.getElementById('outputViewResults').addEventListener('change', () => this.toggleView('results'));
        
        // Listen for troubleshoot WebSocket messages
        if (this.workflowManager.websocket) {
            this.setupWebSocketListener();
        }
    }

    setupWebSocketListener() {
        const originalHandler = this.workflowManager.handleWebSocketMessage.bind(this.workflowManager);
        this.workflowManager.handleWebSocketMessage = (data) => {
            if (data.type === 'troubleshoot_log') {
                this.handleTroubleshootLog(data);
            } else {
                originalHandler(data);
            }
        };
    }

    handleTroubleshootLog(data) {
        const timestamp = new Date(data.timestamp).toLocaleTimeString();
        const logEntry = {
            timestamp: timestamp,
            test: data.test,
            message: data.message,
            level: data.level
        };
        
        this.troubleshootLogs.push(logEntry);
        this.displayLogEntry(logEntry);
        this.updateTestStatus(data.test, data.level, data.message);
    }

    displayLogEntry(logEntry) {
        const logsView = document.getElementById('troubleshootLogsView');
        const logElement = document.createElement('div');
        logElement.className = 'log-entry fade-in';
        
        logElement.innerHTML = `
            <span class="log-timestamp">[${logEntry.timestamp}]</span>
            <span class="log-level-${logEntry.level}">[${logEntry.level.toUpperCase()}]</span>
            <span class="log-test">[${logEntry.test.toUpperCase()}]</span>
            ${logEntry.message}
        `;
        
        logsView.appendChild(logElement);
        logsView.scrollTop = logsView.scrollHeight;
        
        // Keep only last 100 log entries
        while (logsView.children.length > 100) {
            logsView.removeChild(logsView.firstChild);
        }
    }

    updateTestStatus(testType, level, message) {
        const statusElement = document.getElementById(`status-${testType.replace('_', '-')}`);
        if (!statusElement) return;
        
        const icon = statusElement.querySelector('i');
        const badge = statusElement.querySelector('.badge');
        
        if (level === 'success') {
            icon.className = 'fas fa-check-circle text-success me-2';
            badge.className = 'badge bg-success ms-auto';
            badge.textContent = 'Passed';
        } else if (level === 'error') {
            icon.className = 'fas fa-times-circle text-danger me-2';
            badge.className = 'badge bg-danger ms-auto';
            badge.textContent = 'Failed';
        } else if (level === 'warning') {
            icon.className = 'fas fa-exclamation-triangle text-warning me-2';
            badge.className = 'badge bg-warning ms-auto';
            badge.textContent = 'Warning';
        } else if (level === 'info') {
            icon.className = 'fas fa-spinner fa-spin text-primary me-2';
            badge.className = 'badge bg-primary ms-auto';
            badge.textContent = 'Running';
        }
    }

    async runApiHealthTest() {
        this.setTestRunning('api-health');
        this.clearLogsView();
        
        try {
            const response = await fetch(`${this.workflowManager.apiBaseUrl}/troubleshoot/api-health`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const results = await response.json();
                this.currentTestResults['api-health'] = results;
                this.workflowManager.logMessage('API Health Check completed', 'success');
                this.displayTestResults();
            } else {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
        } catch (error) {
            this.workflowManager.logMessage(`API Health Check failed: ${error.message}`, 'error');
            this.updateTestStatus('api-health', 'error', error.message);
        }
    }

    async runDockerOllamaTest() {
        this.setTestRunning('docker-ollama');
        this.clearLogsView();
        
        try {
            const response = await fetch(`${this.workflowManager.apiBaseUrl}/troubleshoot/docker-ollama`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const results = await response.json();
                this.currentTestResults['docker-ollama'] = results;
                this.workflowManager.logMessage('Docker Ollama Test completed', 'success');
                this.displayTestResults();
            } else {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
        } catch (error) {
            this.workflowManager.logMessage(`Docker Ollama Test failed: ${error.message}`, 'error');
            this.updateTestStatus('docker-ollama', 'error', error.message);
        }
    }

    async runModelDebugTest() {
        this.setTestRunning('model-debug');
        this.clearLogsView();
        
        const modelName = document.getElementById('debugModelName').value || 'bge-m3:latest';
        
        try {
            const response = await fetch(`${this.workflowManager.apiBaseUrl}/troubleshoot/model-debug?model_name=${encodeURIComponent(modelName)}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const results = await response.json();
                this.currentTestResults['model-debug'] = results;
                this.workflowManager.logMessage('Model Debug Test completed', 'success');
                this.displayTestResults();
            } else {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
        } catch (error) {
            this.workflowManager.logMessage(`Model Debug Test failed: ${error.message}`, 'error');
            this.updateTestStatus('model-debug', 'error', error.message);
        }
    }

    async runWorkflowModelTest() {
        this.setTestRunning('workflow-model');
        this.clearLogsView();
        
        const config = this.workflowManager.getConfiguration();
        
        try {
            const response = await fetch(`${this.workflowManager.apiBaseUrl}/troubleshoot/workflow-model`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                const results = await response.json();
                this.currentTestResults['workflow-model'] = results;
                this.workflowManager.logMessage('Workflow Model Test completed', 'success');
                this.displayTestResults();
            } else {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
        } catch (error) {
            this.workflowManager.logMessage(`Workflow Model Test failed: ${error.message}`, 'error');
            this.updateTestStatus('workflow-model', 'error', error.message);
        }
    }

    async runLlmDebugTest() {
        this.setTestRunning('llm-debug');
        this.clearLogsView();
        
        const config = this.workflowManager.getConfiguration();
        
        try {
            const response = await fetch(`${this.workflowManager.apiBaseUrl}/troubleshoot/llm-debug`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                const results = await response.json();
                this.currentTestResults['llm-debug'] = results;
                this.workflowManager.logMessage('LLM Manager Debug completed', 'success');
                this.displayTestResults();
            } else {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
        } catch (error) {
            this.workflowManager.logMessage(`LLM Manager Debug failed: ${error.message}`, 'error');
            this.updateTestStatus('llm-debug', 'error', error.message);
        }
    }

    async runCrewWorkflowTest() {
        this.setTestRunning('crew-workflow');
        this.clearLogsView();
        
        const config = this.workflowManager.getConfiguration();
        
        try {
            const response = await fetch(`${this.workflowManager.apiBaseUrl}/troubleshoot/crew-workflow`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                const results = await response.json();
                this.currentTestResults['crew-workflow'] = results;
                this.workflowManager.logMessage('CrewAI Workflow Test completed', 'success');
                this.displayTestResults();
            } else {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
        } catch (error) {
            this.workflowManager.logMessage(`CrewAI Workflow Test failed: ${error.message}`, 'error');
            this.updateTestStatus('crew-workflow', 'error', error.message);
        }
    }

    async runOllamaWorkflowTest() {
        this.setTestRunning('ollama-workflow');
        this.clearLogsView();
        
        const config = this.workflowManager.getConfiguration();
        
        try {
            const response = await fetch(`${this.workflowManager.apiBaseUrl}/troubleshoot/ollama-workflow`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                const results = await response.json();
                this.currentTestResults['ollama-workflow'] = results;
                this.workflowManager.logMessage('Ollama Workflow Test completed', 'success');
                this.displayTestResults();
            } else {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
        } catch (error) {
            this.workflowManager.logMessage(`Ollama Workflow Test failed: ${error.message}`, 'error');
            this.updateTestStatus('ollama-workflow', 'error', error.message);
        }
    }

    async runAllTests() {
        this.workflowManager.logMessage('Starting all troubleshooting tests...', 'info');
        this.clearAllTestStatus();
        
        // Run tests sequentially
        await this.runApiHealthTest();
        await new Promise(resolve => setTimeout(resolve, 1000)); // Small delay between tests
        
        await this.runDockerOllamaTest();
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        await this.runModelDebugTest();
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        await this.runWorkflowModelTest();
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        await this.runLlmDebugTest();
        
        this.workflowManager.logMessage('All troubleshooting tests completed', 'success');
    }

    setTestRunning(testType) {
        this.updateTestStatus(testType.replace('-', '_'), 'info', 'Running test...');
        
        // Add running animation to the test button
        const testButtons = {
            'api-health': 'runApiHealthTest',
            'docker-ollama': 'runDockerOllamaTest',
            'model-debug': 'runModelDebugTest',
            'workflow-model': 'runWorkflowModelTest',
            'llm-debug': 'runLlmDebugTest'
        };
        
        const buttonId = testButtons[testType];
        if (buttonId) {
            const button = document.getElementById(buttonId);
            button.classList.add('test-running');
            button.disabled = true;
            
            // Remove running state after 30 seconds max
            setTimeout(() => {
                button.classList.remove('test-running');
                button.disabled = false;
            }, 30000);
        }
    }

    clearAllTestStatus() {
        const testTypes = ['api-health', 'docker-ollama', 'model-debug', 'workflow-model', 'llm-debug'];
        testTypes.forEach(testType => {
            const statusElement = document.getElementById(`status-${testType}`);
            if (statusElement) {
                const icon = statusElement.querySelector('i');
                const badge = statusElement.querySelector('.badge');
                
                icon.className = 'fas fa-circle text-secondary me-2';
                badge.className = 'badge bg-secondary ms-auto';
                badge.textContent = 'Pending';
            }
        });
        
        this.currentTestResults = {};
        this.troubleshootLogs = [];
    }

    clearLogs() {
        this.troubleshootLogs = [];
        this.clearLogsView();
        this.workflowManager.logMessage('Troubleshooting logs cleared', 'info');
    }

    clearLogsView() {
        const logsView = document.getElementById('troubleshootLogsView');
        logsView.innerHTML = `
            <div class="p-3 text-muted">
                <i class="fas fa-info-circle me-2"></i>
                Test output will appear here...
            </div>
        `;
    }

    toggleView(view) {
        const logsView = document.getElementById('troubleshootLogsView');
        const resultsView = document.getElementById('troubleshootResultsView');
        
        if (view === 'logs') {
            logsView.style.display = 'block';
            resultsView.style.display = 'none';
        } else {
            logsView.style.display = 'none';
            resultsView.style.display = 'block';
            this.displayTestResults();
        }
    }

    displayTestResults() {
        const resultsContent = document.getElementById('troubleshootResultsContent');
        
        if (Object.keys(this.currentTestResults).length === 0) {
            resultsContent.innerHTML = `
                <div class="text-muted text-center py-4">
                    <i class="fas fa-chart-line me-2"></i>
                    No test results available yet
                </div>
            `;
            return;
        }
        
        let html = '';
        
        Object.entries(this.currentTestResults).forEach(([testType, results]) => {
            html += this.createTestResultCard(testType, results);
        });
        
        resultsContent.innerHTML = html;
    }

    createTestResultCard(testType, results) {
        const statusClass = results.overall_status;
        const testName = results.test_name;
        const timestamp = new Date(results.timestamp).toLocaleString();
        
        let summaryHtml = '';
        if (results.summary) {
            summaryHtml = `
                <div class="test-result-summary">
                    <div class="test-result-stat passed">
                        <span class="number">${results.summary.passed}</span>
                        <span class="label">Passed</span>
                    </div>
                    <div class="test-result-stat failed">
                        <span class="number">${results.summary.failed}</span>
                        <span class="label">Failed</span>
                    </div>
                    <div class="test-result-stat warning">
                        <span class="number">${results.summary.warnings || 0}</span>
                        <span class="label">Warnings</span>
                    </div>
                </div>
            `;
        }
        
        let testsHtml = '';
        if (results.tests && results.tests.length > 0) {
            testsHtml = results.tests.map(test => `
                <div class="test-detail-item ${test.status}">
                    <div class="test-detail-header">
                        <span class="test-detail-name">${test.name}</span>
                        <span class="test-detail-status ${test.status}">${test.status.toUpperCase()}</span>
                    </div>
                    <div class="test-detail-message">${test.message}</div>
                    ${test.error ? `<div class="test-detail-error">${test.error}</div>` : ''}
                    ${test.response_time ? `<small class="text-muted">Response time: ${test.response_time}ms</small>` : ''}
                    ${test.models ? `<div class="test-detail-data">Available models: ${JSON.stringify(test.models, null, 2)}</div>` : ''}
                    ${test.model_details ? `<div class="test-detail-data">Model details: ${JSON.stringify(test.model_details, null, 2)}</div>` : ''}
                </div>
            `).join('');
        }
        
        return `
            <div class="test-result-card">
                <div class="test-result-header ${statusClass}">
                    <div>
                        <h6 class="mb-1">${testName}</h6>
                        <small class="text-muted">${timestamp}</small>
                    </div>
                    <span class="badge bg-${statusClass === 'passed' ? 'success' : statusClass === 'failed' ? 'danger' : 'warning'}">${statusClass.toUpperCase()}</span>
                </div>
                <div class="test-result-body">
                    ${summaryHtml}
                    ${testsHtml}
                    ${results.error ? `<div class="test-detail-error">General Error: ${results.error}</div>` : ''}
                </div>
            </div>
        `;
    }

    exportResults() {
        const exportData = {
            timestamp: new Date().toISOString(),
            test_results: this.currentTestResults,
            logs: this.troubleshootLogs,
            system_info: {
                user_agent: navigator.userAgent,
                url: window.location.href,
                configuration: this.workflowManager.getConfiguration()
            }
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `troubleshooting_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        this.workflowManager.showAlert('Troubleshooting results exported successfully!', 'success');
    }

    async loadDebugModels() {
        const debugModelSelect = document.getElementById('debugModelName');
        const refreshBtn = document.getElementById('refreshDebugModelsBtn');
        
        // Show loading state
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Refresh';
        }
        
        debugModelSelect.innerHTML = '<option value="">Loading models...</option>';
        
        try {
            // Get current Ollama URL from configuration
            const ollamaUrl = document.getElementById('ollamaUrl').value || 'http://host.docker.internal:11434';
            
            // Get Ollama models
            const response = await fetch(`${this.workflowManager.apiBaseUrl}/ollama-models?ollama_url=${encodeURIComponent(ollamaUrl)}`);
            if (response.ok) {
                const data = await response.json();
                if (data.error) {
                    debugModelSelect.innerHTML = '<option value="">No models available</option>';
                    this.workflowManager.logMessage(`Failed to load debug models: ${data.error}`, 'warning');
                } else {
                    this.populateDebugModelDropdown(data.models);
                    this.workflowManager.logMessage(`Loaded ${data.models.length} models for debugging`, 'success');
                }
            } else {
                debugModelSelect.innerHTML = '<option value="">Failed to load models</option>';
                this.workflowManager.logMessage('Failed to load debug models from server', 'error');
            }
        } catch (error) {
            debugModelSelect.innerHTML = '<option value="">Error loading models</option>';
            this.workflowManager.logMessage(`Error loading debug models: ${error.message}`, 'error');
        } finally {
            // Reset refresh button
            if (refreshBtn) {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '<i class="fas fa-sync me-1"></i>Refresh';
            }
        }
    }

    populateDebugModelDropdown(models) {
        const debugModelSelect = document.getElementById('debugModelName');
        
        // Clear existing options
        debugModelSelect.innerHTML = '<option value="">Select a model to debug...</option>';
        
        if (models.length === 0) {
            debugModelSelect.innerHTML = '<option value="">No models available</option>';
            return;
        }
        
        // Sort models alphabetically
        const sortedModels = models.sort((a, b) => a.localeCompare(b));
        
        // Add models to dropdown
        sortedModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            
            // Highlight embedding models
            const modelLower = model.toLowerCase();
            if (modelLower.includes('embed') || modelLower.includes('bge') || modelLower.includes('nomic')) {
                option.style.fontWeight = 'bold';
                option.textContent += ' (Embedding Model)';
            }
            
            debugModelSelect.appendChild(option);
        });
        
        // Set default selection to bge-m3:latest if available
        const defaultModel = models.find(m => m.includes('bge-m3')) || models[0];
        if (defaultModel) {
            debugModelSelect.value = defaultModel;
        }
    }

    openTroubleshootingWiki() {
        // Open the troubleshooting wiki in a new window
        const wikiUrl = '/troubleshooting/wiki/index.html';
        const wikiWindow = window.open(wikiUrl, 'troubleshootingWiki', 'width=1200,height=800,scrollbars=yes,resizable=yes');
        
        if (wikiWindow) {
            this.workflowManager.logMessage('Opened troubleshooting wiki', 'info');
        } else {
            this.workflowManager.showAlert('Failed to open wiki. Please check your popup blocker settings.', 'warning');
        }
    }
}

// Initialize troubleshooting manager after workflow manager is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit for workflow manager to initialize
    setTimeout(() => {
        if (workflowManager) {
            window.troubleshootingManager = new TroubleshootingManager(workflowManager);
            // Load debug models after troubleshooting manager is ready
            setTimeout(() => {
                if (window.troubleshootingManager) {
                    window.troubleshootingManager.loadDebugModels();
                }
            }, 500);
        }
    }, 1000);
});
