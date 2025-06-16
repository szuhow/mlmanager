/**
 * Simplified Model Detail UI Manager
 * Fixes flickering indicators and adds epoch navigation
 */

class ModelDetailManager {
    constructor(modelId) {
        this.modelId = modelId;
        this.updateInterval = null;
        this.isTraining = false;
        this.updateIntervalMs = 2000;
        
        this.elements = {
            updateStatus: document.getElementById('update-status'),
            lastUpdateTime: document.getElementById('last-update-time'),
            progressBar: document.querySelector('.progress-bar'),
            batchProgressBar: document.querySelector('.progress-bar.bg-info'),
            stopBtn: document.getElementById('stopTrainingBtn'),
            manualRefreshBtn: document.getElementById('manualRefreshBtn')
        };
        
        this.init();
    }
    
    init() {
        const modelStatus = document.querySelector('[data-model-status]');
        this.isTraining = modelStatus && modelStatus.dataset.modelStatus === 'training';
        
        if (this.isTraining) {
            this.startUpdates();
        }
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to stop training? This cannot be undone.')) {
                    this.stopTraining();
                }
            });
        }
        
        if (this.elements.manualRefreshBtn) {
            this.elements.manualRefreshBtn.addEventListener('click', () => {
                this.manualRefresh();
            });
        }
        
        window.addEventListener('beforeunload', () => {
            this.stopUpdates();
        });
    }
    
    startUpdates() {
        if (this.updateInterval) return;
        
        console.log('Starting model detail updates');
        this.updateProgress();
        this.updateInterval = setInterval(() => {
            this.updateProgress();
        }, this.updateIntervalMs);
        
        this.showLiveIndicator();
    }
    
    stopUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            console.log('Stopped model detail updates');
        }
        this.hideLiveIndicator();
    }
    
    showLiveIndicator() {
        if (this.elements.updateStatus) {
            this.elements.updateStatus.innerHTML = '<i class="fas fa-circle text-success" style="font-size: 0.5rem;"></i> Live';
            this.elements.updateStatus.style.display = 'inline-block';
        }
    }
    
    hideLiveIndicator() {
        if (this.elements.updateStatus) {
            this.elements.updateStatus.style.display = 'none';
        }
    }
    
    setLoadingIndicator() {
        if (this.elements.updateStatus) {
            this.elements.updateStatus.innerHTML = '<i class="fas fa-circle text-warning" style="font-size: 0.5rem; animation: pulse 1s infinite;"></i> Updating...';
        }
    }
    
    setErrorIndicator() {
        if (this.elements.updateStatus) {
            this.elements.updateStatus.innerHTML = '<i class="fas fa-circle text-danger" style="font-size: 0.5rem;"></i> Error';
        }
    }
    
    updateProgress() {
        this.setLoadingIndicator();
        
        fetch(`/ml/model/${this.modelId}/progress/`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    this.handleSuccessfulUpdate(data);
                } else {
                    this.setErrorIndicator();
                }
            })
            .catch(error => {
                console.error('Progress update error:', error);
                this.setErrorIndicator();
            });
    }
    
    handleSuccessfulUpdate(data) {
        this.updateProgressBars(data.progress);
        this.updateMetrics(data.metrics, data.progress);
        this.updateLastUpdateTime();
        this.showLiveIndicator();
        
        if (data.progress && data.progress.current_epoch) {
            this.refreshTrainingImages();
        }
        
        if (data.status_changed) {
            setTimeout(() => {
                location.reload();
            }, 2000);
        }
    }
    
    updateProgressBars(progress) {
        if (!progress) return;
        
        if (this.elements.progressBar) {
            this.elements.progressBar.style.width = progress.percentage + '%';
            this.elements.progressBar.textContent = `Epoch ${progress.current_epoch}/${progress.total_epochs} (${progress.percentage.toFixed(1)}%)`;
            this.elements.progressBar.setAttribute('aria-valuenow', progress.current_epoch);
        }
        
        if (this.elements.batchProgressBar && progress.batch_progress_percentage !== undefined) {
            this.elements.batchProgressBar.style.width = progress.batch_progress_percentage + '%';
            this.elements.batchProgressBar.setAttribute('aria-valuenow', progress.current_batch);
        }
    }
    
    updateMetrics(metrics, progress) {
        if (!metrics) return;
        
        const updateMetric = (id, value) => {
            const element = document.getElementById(id);
            if (element && value !== null && value !== undefined) {
                const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                
                if (element.textContent !== formattedValue) {
                    element.textContent = formattedValue;
                    element.style.backgroundColor = '#e7f3ff';
                    setTimeout(() => {
                        element.style.backgroundColor = '';
                    }, 500);
                }
            }
        };
        
        if (progress) {
            updateMetric('current-epoch', progress.current_epoch);
        }
        updateMetric('train-loss', metrics.train_loss);
        updateMetric('train-dice', metrics.train_dice);
        updateMetric('val-loss', metrics.val_loss);
        updateMetric('val-dice', metrics.val_dice);
        updateMetric('best-val-dice', metrics.best_val_dice);
    }
    
    updateLastUpdateTime() {
        if (this.elements.lastUpdateTime) {
            const now = new Date();
            this.elements.lastUpdateTime.textContent = `Last updated: ${now.toLocaleTimeString()}`;
        }
    }
    
    manualRefresh() {
        if (this.elements.manualRefreshBtn) {
            this.elements.manualRefreshBtn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Refreshing...';
        }
        
        this.updateProgress();
        
        setTimeout(() => {
            if (this.elements.manualRefreshBtn) {
                this.elements.manualRefreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Now';
            }
        }, 1000);
    }
    
    stopTraining() {
        const formData = new FormData();
        formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]').value);
        
        fetch(`/ml/model/${this.modelId}/stop/`, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                if (this.elements.stopBtn) {
                    this.elements.stopBtn.disabled = true;
                    this.elements.stopBtn.textContent = 'Stop Requested';
                }
                this.showAlert('info', 'Stop request sent. Training will stop after current epoch.');
            } else {
                this.showAlert('danger', data.message || 'Failed to stop training');
            }
        })
        .catch(error => {
            console.error('Error stopping training:', error);
            this.showAlert('danger', 'Error stopping training');
        });
    }
    
    refreshTrainingImages() {
        const trainingSamplesCard = document.querySelector('.training-samples-container');
        if (!trainingSamplesCard) return;
        
        fetch(window.location.href, {
            headers: { 'X-Requested-With': 'XMLHttpRequest' }
        })
        .then(response => response.text())
        .then(html => {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const newContainer = doc.querySelector('.training-samples-container');
            
            if (newContainer) {
                trainingSamplesCard.innerHTML = newContainer.innerHTML;
                console.log('Training images refreshed');
            }
        })
        .catch(error => {
            console.error('Error refreshing training images:', error);
        });
    }
    
    // MLflow Registry Management Methods
    async registerModel(modelId) {
        if (!confirm('Register this model in MLflow Model Registry?')) {
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]').value);
            
            const response = await fetch(`/ml/model/${modelId}/registry/register/`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showAlert('success', data.message);
                location.reload();
            } else {
                this.showAlert('danger', data.message);
            }
        } catch (error) {
            console.error('Registry error:', error);
            this.showAlert('danger', 'Failed to register model');
        }
    }
    
    async syncRegistryInfo(modelId) {
        try {
            const response = await fetch(`/ml/model/${modelId}/registry/sync/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                }
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showAlert('success', data.message);
                if (data.changes && data.changes.stage_changed) {
                    const stageBadge = document.getElementById('registry-stage');
                    if (stageBadge) {
                        stageBadge.textContent = data.changes.new_stage || 'None';
                    }
                }
            } else {
                this.showAlert('danger', data.message);
            }
        } catch (error) {
            console.error('Sync error:', error);
            this.showAlert('danger', 'Failed to sync registry info');
        }
    }
    
    // Logs Modal Methods
    openLogsModal() {
        console.log('Opening logs modal...');
        try {
            const modalElement = document.getElementById('logsModal');
            if (!modalElement) {
                console.error('Logs modal element not found');
                return;
            }
            
            if (typeof bootstrap === 'undefined') {
                console.error('Bootstrap is not loaded');
                alert('Unable to open logs modal: Bootstrap library not loaded');
                return;
            }
            
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
            
            // Load logs when modal opens
            this.loadModalLogs();
            console.log('Logs modal opened successfully');
        } catch (error) {
            console.error('Error opening logs modal:', error);
            alert('Error opening logs modal: ' + error.message);
        }
    }
    
    async loadModalLogs() {
        const logsContent = document.getElementById('modal-logs-content');
        const totalLinesSpan = document.getElementById('modal-total-lines');
        
        if (!logsContent) {
            console.error('Modal logs content element not found');
            return;
        }
        
        logsContent.innerHTML = '<div class="text-center py-5"><i class="fas fa-spinner fa-spin fa-2x text-primary"></i><p class="mt-2">Loading logs...</p></div>';
        
        try {
            const response = await fetch(`/ml/model/${this.modelId}/training-log/`, {
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/json',
                },
                credentials: 'same-origin'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            let logsToDisplay = [];
            
            if (data.logs && Array.isArray(data.logs)) {
                logsToDisplay = data.logs.filter(line => line && line.trim() !== '');
            }
            
            // Filter out DEBUG log lines
            logsToDisplay = logsToDisplay.filter(line => {
                const text = typeof line === 'object' && line.content ? line.content : String(line);
                return !/^\d{4}-\d{2}-\d{2}.*\bDEBUG\b/.test(text);
            });
            
            if (logsToDisplay.length > 0) {
                const logHTML = logsToDisplay.map(line => {
                    const text = typeof line === 'object' && line.content ? line.content : String(line);
                    const escapedLine = text.replace(/&/g, '&amp;')
                                           .replace(/</g, '&lt;')
                                           .replace(/>/g, '&gt;');
                    return `<div class="log-entry">${escapedLine}</div>`;
                }).join('');
                logsContent.innerHTML = logHTML;
                if (totalLinesSpan) totalLinesSpan.textContent = logsToDisplay.length;
            } else {
                logsContent.innerHTML = '<div class="text-center py-5"><i class="fas fa-info-circle fa-2x text-warning"></i><p class="mt-2">No logs available yet.</p></div>';
                if (totalLinesSpan) totalLinesSpan.textContent = '0';
            }
            logsContent.scrollTop = logsContent.scrollHeight;
        } catch (error) {
            console.error('Error loading logs:', error);
            logsContent.innerHTML = `<div class="text-center py-5"><i class="fas fa-exclamation-triangle fa-2x text-danger"></i><p class="mt-2">Failed to load logs: ${error.message}</p><p class="text-muted">Please check browser console for details.</p></div>`;
        }
    }
    
    // Alert system
    showAlert(type, message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show mt-3`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert after the header
        const header = document.querySelector('h1').parentElement;
        header.insertAdjacentElement('afterend', alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.remove();
            }
        }, 5000);
    }
}
