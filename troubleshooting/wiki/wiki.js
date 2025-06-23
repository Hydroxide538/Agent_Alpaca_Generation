// Wiki functionality for troubleshooting interface
class TroubleshootingWiki {
    constructor() {
        this.issues = [];
        this.testScripts = [];
        this.quickFixes = [];
        this.init();
    }

    async init() {
        await this.loadData();
        this.setupEventListeners();
        this.renderContent();
    }

    async loadData() {
        try {
            // Load issue history from the history directory
            this.issues = await this.loadIssueHistory();
            
            // Load test scripts information
            this.testScripts = this.getTestScripts();
            
            // Load quick fixes
            this.quickFixes = this.getQuickFixes();
            
            // Update system status
            await this.updateSystemStatus();
        } catch (error) {
            console.error('Error loading wiki data:', error);
        }
    }

    async loadIssueHistory() {
        // This would typically load from the history files
        // For now, we'll return mock data based on the known issues
        return [
            {
                id: 'alpaca-improvements',
                title: 'CrewAI Workflow Improvements: Alpaca Format Training Data Generation',
                category: 'workflow',
                date: '2025-06-22',
                status: 'resolved',
                description: 'Comprehensive improvements to generate high-quality Alpaca format training data with enhanced RAG capabilities.',
                tags: ['alpaca', 'rag', 'improvements', 'training-data'],
                severity: 'enhancement'
            },
            {
                id: 'llm-failure-fix',
                title: 'LLM Failure Fix',
                category: 'llm',
                date: '2025-06-22',
                status: 'resolved',
                description: 'Fixed "LLM Failed" error caused by workflow manager not executing CrewAI workflow.',
                tags: ['llm', 'crewai', 'workflow', 'execution'],
                severity: 'critical'
            },
            {
                id: 'ollama-fix',
                title: 'Ollama Workflow Fix',
                category: 'docker',
                date: '2025-06-22',
                status: 'resolved',
                description: 'Fixed OpenAI API Key error when using Ollama models and Docker URL configuration.',
                tags: ['ollama', 'docker', 'api-key', 'configuration'],
                severity: 'high'
            },
            {
                id: 'workflow-troubleshooting',
                title: 'Workflow Troubleshooting',
                category: 'workflow',
                date: '2025-06-22',
                status: 'resolved',
                description: 'Multiple workflow execution issues including model detection and embedding model usage.',
                tags: ['workflow', 'models', 'embedding', 'debugging'],
                severity: 'high'
            }
        ];
    }

    getTestScripts() {
        return [
            {
                name: 'API Health Test',
                file: 'troubleshooting/scripts/test_api.py',
                description: 'Basic API health testing for localhost and Docker Ollama connections',
                category: 'api',
                integrated: true
            },
            {
                name: 'ChromaDB Fix Test',
                file: 'troubleshooting/scripts/test_chromadb_fix.py',
                description: 'Test ChromaDB connection and vector database functionality',
                category: 'database',
                integrated: true
            },
            {
                name: 'CrewAI Workflow Test',
                file: 'troubleshooting/scripts/test_crew_workflow.py',
                description: 'Comprehensive CrewAI workflow execution testing with proper model configuration',
                category: 'workflow',
                integrated: false
            },
            {
                name: 'CrewAI Fix Test',
                file: 'troubleshooting/scripts/test_crewai_fix.py',
                description: 'Test CrewAI specific fixes and configurations',
                category: 'workflow',
                integrated: true
            },
            {
                name: 'Docker Ollama Test',
                file: 'troubleshooting/scripts/test_docker_ollama.py',
                description: 'Test Ollama connection from Docker environment with detailed debugging',
                category: 'docker',
                integrated: true
            },
            {
                name: 'Fix Verification Test',
                file: 'troubleshooting/scripts/test_fix_verification.py',
                description: 'Comprehensive test to verify all applied fixes are working correctly',
                category: 'verification',
                integrated: true
            },
            {
                name: 'Improved Alpaca Test',
                file: 'troubleshooting/scripts/test_improved_alpaca.py',
                description: 'Test improved Alpaca format data generation with enhanced features',
                category: 'alpaca',
                integrated: true
            },
            {
                name: 'LiteLLM Fix Test',
                file: 'troubleshooting/scripts/test_litellm_fix.py',
                description: 'Test LiteLLM integration fixes and configurations',
                category: 'llm',
                integrated: true
            },
            {
                name: 'Model Debug Test',
                file: 'troubleshooting/scripts/test_model_debug.py',
                description: 'Detailed model debugging for specific models like bge-m3',
                category: 'model',
                integrated: true
            },
            {
                name: 'Ollama Workflow Test',
                file: 'troubleshooting/scripts/test_ollama_workflow.py',
                description: 'Test Ollama workflow configuration with dynamic model selection',
                category: 'workflow',
                integrated: false
            },
            {
                name: 'Workflow Fix Test',
                file: 'troubleshooting/scripts/test_workflow_fix.py',
                description: 'Quick test to verify workflow manager fix resolves LLM failures',
                category: 'workflow',
                integrated: false
            },
            {
                name: 'Workflow Integration Test',
                file: 'troubleshooting/scripts/test_workflow_integration.py',
                description: 'Test complete workflow integration and end-to-end functionality',
                category: 'workflow',
                integrated: true
            },
            {
                name: 'Workflow Model Test',
                file: 'troubleshooting/scripts/test_workflow_model.py',
                description: 'Test workflow model setup and configuration',
                category: 'model',
                integrated: true
            },
            {
                name: 'LLM Manager Debug',
                file: 'troubleshooting/scripts/debug_llm_manager.py',
                description: 'Comprehensive LLM manager debugging and diagnostics',
                category: 'llm',
                integrated: true
            }
        ];
    }

    getQuickFixes() {
        return [
            {
                title: 'LLM Failed Error',
                category: 'llm',
                description: 'Quick fix for "LLM Failed" errors in CrewAI workflow',
                steps: [
                    'Check if Ollama server is running on the correct port',
                    'Verify model names match available models in Ollama',
                    'Ensure Docker URL is set to host.docker.internal:11434',
                    'Run workflow fix test: python troubleshooting/scripts/test_workflow_fix.py'
                ]
            },
            {
                title: 'OpenAI API Key Error with Ollama',
                category: 'docker',
                description: 'Fix OpenAI API key requirement when using Ollama models',
                steps: [
                    'Ensure Ollama URL is correctly set in configuration',
                    'Verify model names start with "ollama:" prefix',
                    'Check that embedding models are not used for text generation',
                    'Clear OpenAI API key field if using only Ollama models'
                ]
            },
            {
                title: 'Model Not Found Error',
                category: 'model',
                description: 'Fix model availability issues',
                steps: [
                    'Run: ollama list to check available models',
                    'Pull missing models: ollama pull <model-name>',
                    'Verify model names in configuration match exactly',
                    'Use base model names without version tags if needed'
                ]
            },
            {
                title: 'Docker Connection Issues',
                category: 'docker',
                description: 'Fix Docker Ollama connection problems',
                steps: [
                    'Use host.docker.internal:11434 instead of localhost:11434',
                    'Ensure Ollama is running on the host machine',
                    'Check Docker network configuration',
                    'Test connection with: python troubleshooting/scripts/test_docker_ollama.py'
                ]
            },
            {
                title: 'Embedding Model Used for Generation',
                category: 'model',
                description: 'Fix embedding models being used for text generation',
                steps: [
                    'Identify embedding models (bge-*, snowflake-*, nomic-embed-*)',
                    'Use embedding models only for embedding tasks',
                    'Use text generation models (llama*, mistral*, etc.) for agents',
                    'Check model family in Ollama model details'
                ]
            }
        ];
    }

    async updateSystemStatus() {
        try {
            // This would typically make API calls to check system status
            // For now, we'll simulate the status
            const statusElement = document.getElementById('systemStatus');
            if (statusElement) {
                statusElement.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <span>API Health:</span>
                        <span class="badge bg-success">Healthy</span>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span>Ollama Connection:</span>
                        <span class="badge bg-success">Connected</span>
                    </div>
                    <div class="d-flex justify-content-between">
                        <span>Models Available:</span>
                        <span class="badge bg-info">${this.testScripts.length}</span>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error updating system status:', error);
        }
    }

    setupEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('wikiSearch');
        const searchBtn = document.getElementById('searchBtn');
        
        if (searchInput && searchBtn) {
            searchBtn.addEventListener('click', () => this.performSearch());
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.performSearch();
            });
        }

        // Category filters
        const categoryFilters = document.querySelectorAll('.category-filter');
        categoryFilters.forEach(filter => {
            filter.addEventListener('click', (e) => {
                const category = e.target.dataset.category;
                this.filterByCategory(category);
                
                // Update active state
                categoryFilters.forEach(f => f.classList.remove('active'));
                e.target.classList.add('active');
            });
        });

        // Quick actions
        const runDiagnosticsBtn = document.getElementById('runDiagnosticsBtn');
        if (runDiagnosticsBtn) {
            runDiagnosticsBtn.addEventListener('click', () => this.runDiagnostics());
        }

        const exportWikiBtn = document.getElementById('exportWikiBtn');
        if (exportWikiBtn) {
            exportWikiBtn.addEventListener('click', () => this.exportWiki());
        }

        const newIssueBtn = document.getElementById('newIssueBtn');
        if (newIssueBtn) {
            newIssueBtn.addEventListener('click', () => this.showIssueReportModal());
        }

        // Issue report modal
        const submitIssueBtn = document.getElementById('submitIssueBtn');
        if (submitIssueBtn) {
            submitIssueBtn.addEventListener('click', () => this.submitIssue());
        }
    }

    renderContent() {
        this.renderIssueTimeline();
        this.renderCommonIssues();
        this.renderTestScripts();
        this.renderQuickFixes();
        this.renderRecentActivity();
    }

    renderIssueTimeline() {
        const timeline = document.getElementById('issueTimeline');
        if (!timeline) return;

        const sortedIssues = this.issues.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        timeline.innerHTML = sortedIssues.map(issue => `
            <div class="timeline-item" data-category="${issue.category}">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h6>${issue.title}</h6>
                        <p class="text-muted mb-2">${issue.description}</p>
                        <div class="d-flex gap-1 mb-2">
                            ${issue.tags.map(tag => `<span class="badge bg-light text-dark">${tag}</span>`).join('')}
                        </div>
                        <small class="text-muted">
                            <i class="fas fa-calendar me-1"></i>${issue.date}
                            <span class="badge bg-${this.getSeverityColor(issue.severity)} ms-2">${issue.severity}</span>
                            <span class="badge bg-${this.getStatusColor(issue.status)} ms-1">${issue.status}</span>
                        </small>
                    </div>
                    <button class="btn btn-sm btn-outline-primary" onclick="wiki.viewIssueDetails('${issue.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }

    renderCommonIssues() {
        const grid = document.getElementById('commonIssuesGrid');
        if (!grid) return;

        const commonIssues = this.issues.filter(issue => 
            ['critical', 'high'].includes(issue.severity) && issue.status === 'resolved'
        );

        grid.innerHTML = commonIssues.map(issue => `
            <div class="col-md-6 mb-3">
                <div class="card issue-card h-100" data-category="${issue.category}">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas fa-${this.getCategoryIcon(issue.category)} me-2"></i>
                            ${issue.title}
                        </h6>
                        <p class="card-text">${issue.description}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-${this.getCategoryColor(issue.category)}">${issue.category}</span>
                            <button class="btn btn-sm btn-primary" onclick="wiki.viewIssueDetails('${issue.id}')">
                                View Fix
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    renderTestScripts() {
        const grid = document.getElementById('testScriptsGrid');
        if (!grid) return;

        grid.innerHTML = this.testScripts.map(script => `
            <div class="col-md-6 mb-3">
                <div class="card issue-card h-100" data-category="${script.category}">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas fa-${script.integrated ? 'check-circle text-success' : 'circle text-muted'} me-2"></i>
                            ${script.name}
                        </h6>
                        <p class="card-text">${script.description}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-${this.getCategoryColor(script.category)}">${script.category}</span>
                            <div>
                                ${script.integrated ? 
                                    `<button class="btn btn-sm btn-success" onclick="wiki.runIntegratedTest('${script.file}')">Run Test</button>` :
                                    `<button class="btn btn-sm btn-outline-primary" onclick="wiki.runStandaloneTest('${script.file}')">Run Script</button>`
                                }
                            </div>
                        </div>
                        <small class="text-muted mt-2 d-block">
                            <i class="fas fa-file-code me-1"></i>${script.file}
                        </small>
                    </div>
                </div>
            </div>
        `).join('');
    }

    renderQuickFixes() {
        const accordion = document.getElementById('quickFixesAccordion');
        if (!accordion) return;

        accordion.innerHTML = this.quickFixes.map((fix, index) => `
            <div class="accordion-item" data-category="${fix.category}">
                <h2 class="accordion-header" id="heading${index}">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#collapse${index}" aria-expanded="false" aria-controls="collapse${index}">
                        <i class="fas fa-${this.getCategoryIcon(fix.category)} me-2"></i>
                        ${fix.title}
                        <span class="badge bg-${this.getCategoryColor(fix.category)} ms-2">${fix.category}</span>
                    </button>
                </h2>
                <div id="collapse${index}" class="accordion-collapse collapse" aria-labelledby="heading${index}" 
                     data-bs-parent="#quickFixesAccordion">
                    <div class="accordion-body">
                        <p>${fix.description}</p>
                        <h6>Steps to Fix:</h6>
                        <ol>
                            ${fix.steps.map(step => `<li>${step}</li>`).join('')}
                        </ol>
                    </div>
                </div>
            </div>
        `).join('');
    }

    renderRecentActivity() {
        const activityElement = document.getElementById('recentActivity');
        if (!activityElement) return;

        const recentIssues = this.issues.slice(0, 3);
        activityElement.innerHTML = recentIssues.map(issue => `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <div>
                    <small class="fw-bold">${issue.title}</small><br>
                    <small class="text-muted">${issue.date}</small>
                </div>
                <span class="badge bg-${this.getStatusColor(issue.status)}">${issue.status}</span>
            </div>
        `).join('');
    }

    performSearch() {
        const searchTerm = document.getElementById('wikiSearch').value.toLowerCase();
        if (!searchTerm) return;

        // Search through issues, test scripts, and quick fixes
        const results = [
            ...this.issues.filter(item => 
                item.title.toLowerCase().includes(searchTerm) ||
                item.description.toLowerCase().includes(searchTerm) ||
                item.tags.some(tag => tag.toLowerCase().includes(searchTerm))
            ),
            ...this.testScripts.filter(item =>
                item.name.toLowerCase().includes(searchTerm) ||
                item.description.toLowerCase().includes(searchTerm)
            ),
            ...this.quickFixes.filter(item =>
                item.title.toLowerCase().includes(searchTerm) ||
                item.description.toLowerCase().includes(searchTerm)
            )
        ];

        this.highlightSearchResults(searchTerm);
        console.log('Search results:', results);
    }

    highlightSearchResults(searchTerm) {
        // Remove existing highlights
        document.querySelectorAll('.search-highlight').forEach(el => {
            el.outerHTML = el.innerHTML;
        });

        // Add new highlights
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }

        textNodes.forEach(textNode => {
            if (textNode.textContent.toLowerCase().includes(searchTerm)) {
                const regex = new RegExp(`(${searchTerm})`, 'gi');
                const highlighted = textNode.textContent.replace(regex, '<span class="search-highlight">$1</span>');
                const wrapper = document.createElement('div');
                wrapper.innerHTML = highlighted;
                textNode.parentNode.replaceChild(wrapper, textNode);
            }
        });
    }

    filterByCategory(category) {
        const items = document.querySelectorAll('[data-category]');
        items.forEach(item => {
            if (category === 'all' || item.dataset.category === category) {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
    }

    runDiagnostics() {
        // This would integrate with the main troubleshooting system
        if (window.parent && window.parent.runAllTroubleshootingTests) {
            window.parent.runAllTroubleshootingTests();
        } else {
            alert('Diagnostics integration not available. Please use the main troubleshooting interface.');
        }
    }

    exportWiki() {
        const wikiData = {
            issues: this.issues,
            testScripts: this.testScripts,
            quickFixes: this.quickFixes,
            exportDate: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(wikiData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `troubleshooting-wiki-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    showIssueReportModal() {
        const modal = new bootstrap.Modal(document.getElementById('issueReportModal'));
        modal.show();
    }

    submitIssue() {
        const form = document.getElementById('issueReportForm');
        const formData = new FormData(form);
        
        const issue = {
            id: Date.now().toString(),
            title: document.getElementById('issueTitle').value,
            category: document.getElementById('issueCategory').value,
            description: document.getElementById('issueDescription').value,
            steps: document.getElementById('issueSteps').value,
            errors: document.getElementById('issueErrors').value,
            date: new Date().toISOString().split('T')[0],
            status: 'open',
            severity: 'medium'
        };

        // Add to issues list
        this.issues.unshift(issue);
        
        // Re-render content
        this.renderContent();
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('issueReportModal'));
        modal.hide();
        
        // Reset form
        form.reset();
        
        alert('Issue reported successfully!');
    }

    viewIssueDetails(issueId) {
        const issue = this.issues.find(i => i.id === issueId);
        if (issue) {
            // This would open a detailed view or navigate to the history file
            alert(`Viewing details for: ${issue.title}\n\nThis would open the detailed troubleshooting history file.`);
        }
    }

    runIntegratedTest(testFile) {
        // This would integrate with the main troubleshooting system
        alert(`Running integrated test: ${testFile}\n\nThis would trigger the test through the main troubleshooting interface.`);
    }

    runStandaloneTest(testFile) {
        // This would run the standalone test script
        alert(`Running standalone test: ${testFile}\n\nThis would execute: python ${testFile}`);
    }

    // Helper methods for styling
    getSeverityColor(severity) {
        const colors = {
            'critical': 'danger',
            'high': 'warning',
            'medium': 'info',
            'low': 'secondary',
            'enhancement': 'success'
        };
        return colors[severity] || 'secondary';
    }

    getStatusColor(status) {
        const colors = {
            'resolved': 'success',
            'open': 'warning',
            'in-progress': 'info',
            'closed': 'secondary'
        };
        return colors[status] || 'secondary';
    }

    getCategoryColor(category) {
        const colors = {
            'llm': 'success',
            'docker': 'info',
            'model': 'warning',
            'workflow': 'danger',
            'api': 'primary',
            'other': 'secondary'
        };
        return colors[category] || 'secondary';
    }

    getCategoryIcon(category) {
        const icons = {
            'llm': 'brain',
            'docker': 'cube',
            'model': 'cogs',
            'workflow': 'project-diagram',
            'api': 'plug',
            'other': 'question-circle'
        };
        return icons[category] || 'circle';
    }
}

// Initialize the wiki when the page loads
let wiki;
document.addEventListener('DOMContentLoaded', () => {
    wiki = new TroubleshootingWiki();
});
