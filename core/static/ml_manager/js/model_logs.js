class ModelLogsManager {
    constructor(modelId) {
        this.modelId = modelId;
        this.currentFilters = {
            type: 'all',
            search: ''
        };
        this.autoRefreshInterval = null;
        this.initializeEventListeners();
        this.loadLogs();
    }

    initializeEventListeners() {
        // Log type filter buttons
        const logViewButtons = document.querySelectorAll('input[name="logView"]');
        logViewButtons.forEach(button => {
            button.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.currentFilters.type = e.target.id.replace('logView', '').toLowerCase();
                    // Map button IDs to filter types that match Django expectations
                    const filterMap = {
                        'all': 'all',
                        'epoch': 'epochs',      // JS: epoch button -> Django: epochs filter
                        'batch': 'batches',     // JS: batch button -> Django: batches filter  
                        'metrics': 'metrics'    // JS: metrics button -> Django: metrics filter
                    };
                    this.currentFilters.type = filterMap[this.currentFilters.type] || 'all';
                    this.loadLogs();
                }
            });
        });

        // Search filter
        const searchInput = document.getElementById('log-search');
        if (searchInput) {
            let searchTimeout;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.currentFilters.search = e.target.value;
                    this.loadLogs();
                }, 300);
            });
        }

        // Refresh button
        const refreshBtn = document.getElementById('refresh-logs');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadLogs();
            });
        }

        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('auto-refresh-logs');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startAutoRefresh();
                } else {
                    this.stopAutoRefresh();
                }
            });
        }

        // Clear search button
        const clearSearchBtn = document.getElementById('clear-search');
        if (clearSearchBtn) {
            clearSearchBtn.addEventListener('click', () => {
                if (searchInput) {
                    searchInput.value = '';
                    this.currentFilters.search = '';
                    this.loadLogs();
                }
            });
        }
    }

    async loadLogs() {
        const logsContainer = document.getElementById('training-logs-container');
        if (!logsContainer) return;

        // Show loading indicator
        this.showLoading();

        try {
            const params = new URLSearchParams(this.currentFilters);
            const response = await fetch(`/ml/model/${this.modelId}/logs/?${params}`, {
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.renderLogs(data.logs);
            this.updateLogStats(data);

        } catch (error) {
            console.error('Error loading logs:', error);
            this.showError('Failed to load logs: ' + error.message);
        }
    }

    renderLogs(logs) {
        // Update the specific log view containers instead of just training-logs-container
        const currentFilter = this.currentFilters.type;
        let targetContainerId;
        
        // Map filter type to target container
        switch(currentFilter) {
            case 'epochs':
                targetContainerId = 'epoch-logs';
                break;
            case 'batches':
                targetContainerId = 'batch-logs';
                break;
            case 'metrics':
                targetContainerId = 'metrics-logs';
                break;
            default:
                targetContainerId = 'all-logs';
        }
        
        const targetContainer = document.getElementById(targetContainerId);
        if (!targetContainer) {
            console.warn('Target container not found:', targetContainerId);
            return;
        }

        if (logs.length === 0) {
            targetContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> No logs found matching the current filters.
                </div>
            `;
            return;
        }

        // Clear existing content and render new logs
        const logSection = targetContainer.querySelector('.log-section') || targetContainer;
        
        const logsHtml = logs.map((log, index) => {
            // Handle both string and object log formats defensively
            let logData;
            if (typeof log === 'string') {
                logData = {
                    line_number: index + 1,
                    content: log,
                    timestamp: '',
                    level: 'INFO'
                };
            } else if (typeof log === 'object' && log !== null) {
                logData = {
                    line_number: log.line_number || index + 1,
                    content: log.content || String(log),
                    timestamp: log.timestamp || '',
                    level: log.level || 'INFO'
                };
            } else {
                logData = {
                    line_number: index + 1,
                    content: String(log),
                    timestamp: '',
                    level: 'INFO'
                };
            }

            // Determine CSS class based on content and filter type
            let cssClass = 'log-entry';
            if (currentFilter === 'epochs' || logData.content.includes('[EPOCH]')) {
                cssClass += ' epoch-log';
            } else if (currentFilter === 'batches' || logData.content.includes('[TRAIN]') || logData.content.includes('[VAL]')) {
                cssClass += logData.content.includes('[VAL]') ? ' validation-log' : ' batch-log';
            } else if (currentFilter === 'metrics' || logData.content.includes('[METRICS]')) {
                cssClass += ' metrics-log';
            } else if (logData.content.includes('[CONFIG]')) {
                cssClass += ' config-log';
            } else if (logData.content.includes('[MODEL]')) {
                cssClass += ' model-log';
            }

            return `
                <div class="${cssClass}">
                    ${logData.content}
                </div>
            `;
        }).join('');

        logSection.innerHTML = logsHtml;
        
        // Auto-scroll to bottom
        if (logSection.scrollTo) {
            logSection.scrollTop = logSection.scrollHeight;
        }
        
        // Update filter info display
        this.updateFilterInfo();
    }

    updateFilterInfo() {
        const filterInfo = document.getElementById('filter-info');
        if (filterInfo) {
            const searchText = this.currentFilters.search ? ` | Search: "${this.currentFilters.search}"` : '';
            filterInfo.textContent = `Type: ${this.currentFilters.type}${searchText}`;
        }
    }

    showLoading() {
        const currentFilter = this.currentFilters.type;
        let targetContainerId;
        
        // Map filter type to target container
        switch(currentFilter) {
            case 'epochs':
                targetContainerId = 'epoch-logs';
                break;
            case 'batches':
                targetContainerId = 'batch-logs';
                break;
            case 'metrics':
                targetContainerId = 'metrics-logs';
                break;
            default:
                targetContainerId = 'all-logs';
        }
        
        const targetContainer = document.getElementById(targetContainerId);
        if (targetContainer) {
            targetContainer.innerHTML = `
                <div class="text-center p-4">
                    <i class="fas fa-spinner fa-spin fa-2x text-primary"></i>
                    <p class="mt-2">Loading logs...</p>
                </div>
            `;
        }
    }

    showError(message) {
        const currentFilter = this.currentFilters.type;
        let targetContainerId;
        
        // Map filter type to target container
        switch(currentFilter) {
            case 'epochs':
                targetContainerId = 'epoch-logs';
                break;
            case 'batches':
                targetContainerId = 'batch-logs';
                break;
            case 'metrics':
                targetContainerId = 'metrics-logs';
                break;
            default:
                targetContainerId = 'all-logs';
        }
        
        const targetContainer = document.getElementById(targetContainerId);
        if (targetContainer) {
            targetContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i> ${message}
                </div>
            `;
        }
        console.error('ModelLogsManager Error:', message);
    }

    updateLogStats(data) {
        // Update log count in stats
        const logStats = document.getElementById('log-stats');
        if (logStats) {
            const totalElement = logStats.querySelector('#total-lines');
            if (totalElement) {
                totalElement.textContent = data.total_lines || data.logs?.length || 0;
            }
        }

        // Update filter info
        const filterInfo = document.getElementById('filter-info');
        if (filterInfo) {
            let info = `Type: ${this.currentFilters.type}`;
            if (this.currentFilters.search) {
                info += `, Search: "${this.currentFilters.search}"`;
            }
            filterInfo.textContent = info;
        }
    }

    getLevelBadgeClass(level) {
        switch (level.toUpperCase()) {
            case 'ERROR': return 'danger';
            case 'WARNING': return 'warning';
            case 'INFO': return 'info';
            case 'DEBUG': return 'secondary';
            case 'EPOCH': return 'primary';
            case 'TRAINING': return 'success';
            case 'METRICS': return 'info';
            default: return 'light';
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    startAutoRefresh() {
        this.stopAutoRefresh(); // Clear any existing interval
        this.autoRefreshInterval = setInterval(() => {
            this.loadLogs();
        }, 5000); // Refresh every 5 seconds
        
        console.log('ModelLogsManager: Auto-refresh started');
    }

    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
            console.log('ModelLogsManager: Auto-refresh stopped');
        }
    }
}

// Initialize when page loads if model detail page
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on a model detail page with logs
    const modelContainer = document.querySelector('[data-model-id]');
    const logsContainer = document.getElementById('training-logs-container');
    
    if (modelContainer && logsContainer) {
        const modelId = modelContainer.dataset.modelId;
        console.log('ModelLogsManager: Initializing for model', modelId);
        window.modelLogsManager = new ModelLogsManager(modelId);
        
        // Auto-start refresh if model is training
        const modelStatus = modelContainer.dataset.modelStatus;
        if (modelStatus === 'training') {
            const autoRefreshToggle = document.getElementById('auto-refresh-logs');
            if (autoRefreshToggle) {
                autoRefreshToggle.checked = true;
                window.modelLogsManager.startAutoRefresh();
            }
        }
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (window.modelLogsManager) {
        window.modelLogsManager.stopAutoRefresh();
    }
});
