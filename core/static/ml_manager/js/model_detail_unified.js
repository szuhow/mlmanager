/**
 * Unified Model Detail UI Manager
 * Consolidates all update mechanisms to prevent flickering and conflicts
 */

// Debug: Test if JS file is loaded
console.log('🚀 model_detail_unified.js LOADED at', new Date().toISOString());
console.log('🔍 Browser info:', navigator.userAgent);
console.log('🔍 Location:', window.location.href);

// CSRF Token handling for AJAX requests
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrftoken = getCookie('csrftoken');

// Setup CSRF for all AJAX requests
function setupCSRF() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]')?.value || csrftoken;
    return {
        'X-CSRFToken': token,
        'Content-Type': 'application/json',
    };
}

class ModelDetailManager {
    constructor(modelId) {
        console.log(`ModelDetailManager: Initializing for model ${modelId}`);
        this.modelId = modelId;
        this.updateInterval = null;
        this.isTraining = false;
        this.lastUpdateTime = null;
        this.updateIntervalMs = 2000; // 2 seconds
        
        // UI elements cache - using more specific selectors
        this.elements = {
            updateStatus: document.getElementById('update-status'),
            lastUpdateTime: document.getElementById('last-update-time'),
            progressBar: document.querySelector('#training-progress .progress-enhanced .progress-bar'),
            batchProgressBar: document.querySelector('#training-progress .progress-sm .progress-bar.bg-info'),
            batchText: document.querySelector('#training-progress small.text-muted.mb-3.d-block'),
            stopBtn: document.getElementById('stopTrainingBtn'),
            manualRefreshBtn: document.getElementById('manualRefreshBtn')
        };
        
        // Debug: Log element availability
        console.log('ModelDetailManager: Element check:');
        console.log('  progressBar:', !!this.elements.progressBar, this.elements.progressBar);
        console.log('  batchProgressBar:', !!this.elements.batchProgressBar, this.elements.batchProgressBar);
        console.log('  batchText:', !!this.elements.batchText, this.elements.batchText);
        console.log('  updateStatus:', !!this.elements.updateStatus);
        console.log('  lastUpdateTime:', !!this.elements.lastUpdateTime);
        console.log('  manualRefreshBtn:', !!this.elements.manualRefreshBtn);
        
        this.init();
        
        // Debug: Confirm initialization completed
        console.log('🎯 ModelDetailManager initialization completed');
        console.log('🎯 this.isTraining:', this.isTraining);
        console.log('🎯 this.updateInterval:', this.updateInterval);
    }
    
    init() {
        // Check if model is training or recently started
        const modelStatus = document.querySelector('[data-model-status]');
        const statusValue = modelStatus?.dataset.modelStatus;
        
        // Debug: Log status detection
        console.log('ModelDetailManager: Status detection:');
        console.log('  modelStatus element:', !!modelStatus);
        console.log('  statusValue:', statusValue);
        console.log('  dataset:', modelStatus?.dataset);
        
        // Start updates for training, pending, or loading states (covers recently started training)
        this.isTraining = statusValue && ['training', 'pending', 'loading'].includes(statusValue);
        
        console.log('  isTraining:', this.isTraining);
        
        // Preload logs data in the background
        this.preloadLogs();
        
        // Always start updates for all model states to detect status changes
        // This ensures "Training Pending" will transition automatically
        console.log(`ModelDetailManager: Starting updates for model status: ${statusValue}`);
        this.startUpdates();
        
        if (this.isTraining) {
            this.showLiveIndicator();
        } else {
            console.log(`ModelDetailManager: Model status is ${statusValue}, still watching for status changes`);
            
            // Smart detection: Check if this might be a recently created model that will start training soon
            const modelElement = document.querySelector('[data-model-id]');
            const modelCreatedAt = modelElement?.dataset.modelCreatedAt;
            
            if (modelCreatedAt) {
                const createdTime = new Date(modelCreatedAt);
                const now = new Date();
                const minutesSinceCreation = (now - createdTime) / (1000 * 60);
                
                // If model was created within the last 5 minutes, poll for status changes
                if (minutesSinceCreation < 5) {
                    console.log(`ModelDetailManager: Model created ${minutesSinceCreation.toFixed(1)} minutes ago, watching for training start`);
                    this.startTrainingWatch();
                }
            }
        }
        
        this.setupEventListeners();
        
        // Preload logs immediately for faster access
        this.preloadLogs();
    }
    
    preloadLogs() {
        // Preload logs data in the background so it's ready when modal opens
        console.log('ModelDetailManager: Preloading logs data');
        const url = `/ml/model/${this.modelId}/logs/`;
        
        // Store logs data in this.logsData for faster display when modal opens
        fetch(url, {
            method: 'GET',
            headers: setupCSRF(),
            credentials: 'same-origin'
        })
            .then(response => response.json())
            .then(data => {
                this.logsData = data;
                console.log('ModelDetailManager: Logs preloaded successfully');
            })
            .catch(error => {
                console.error('Error preloading logs:', error);
            });
    }
    
    startTrainingWatch() {
        // Watch for training to start for up to 2 minutes
        let watchAttempts = 0;
        const maxAttempts = 24; // 24 * 5 seconds = 2 minutes
        
        const watchInterval = setInterval(() => {
            watchAttempts++;
            
            fetch(`/ml/model/${this.modelId}/progress/`, {
                method: 'GET',
                headers: setupCSRF(),
                credentials: 'same-origin'
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success' && data.model_status) {
                        const currentStatus = data.model_status;
                        
                        if (['training', 'loading'].includes(currentStatus)) {
                            console.log(`ModelDetailManager: Training started! Status: ${currentStatus}`);
                            clearInterval(watchInterval);
                            this.isTraining = true;
                            this.startUpdates();
                            this.showLiveIndicator();
                        } else if (watchAttempts >= maxAttempts) {
                            console.log('ModelDetailManager: Training watch timeout, stopping');
                            clearInterval(watchInterval);
                        }
                    }
                })
                .catch(error => {
                    console.error('Training watch error:', error);
                    if (watchAttempts >= maxAttempts) {
                        clearInterval(watchInterval);
                    }
                });
        }, 5000); // Check every 5 seconds
    }
    
    setupEventListeners() {
        // Stop training button
        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener('click', () => {
                this.stopTraining();
            });
        }
        
        // Manual refresh button
        if (this.elements.manualRefreshBtn) {
            this.elements.manualRefreshBtn.addEventListener('click', () => {
                this.manualRefresh();
            });
        }
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            this.stopUpdates();
        });
    }
    
    startUpdates() {
        if (this.updateInterval) return; // Already running
        
        console.log('ModelDetailManager: Starting updates');
        this.updateProgress(); // Initial update
        this.updateInterval = setInterval(() => {
            this.updateProgress();
        }, this.updateIntervalMs);
        
        this.showLiveIndicator();
    }
    
    stopUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            console.log('ModelDetailManager: Stopped updates');
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
        
        const url = `/ml/model/${this.modelId}/progress/`;
        console.log('ModelDetailManager: Fetching progress from', url);
        
        fetch(url, {
            method: 'GET',
            headers: setupCSRF(),
            credentials: 'same-origin'
        })
            .then(response => {
                console.log('ModelDetailManager: Progress response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('ModelDetailManager: Progress data received:', data);
                if (data.status === 'success') {
                    this.handleSuccessfulUpdate(data);
                } else {
                    console.error('ModelDetailManager: API returned error:', data.message || 'Unknown error');
                    this.setErrorIndicator();
                }
            })
            .catch(error => {
                console.error('ModelDetailManager: Progress update error:', error);
                this.setErrorIndicator();
            });
    }
    
    handleSuccessfulUpdate(data) {
        // Check for status changes 
        const currentModelStatus = data.model_status;
        const modelElement = document.querySelector('[data-model-status]');
        const currentUIStatus = modelElement?.dataset.modelStatus;
        
        console.log('ModelDetailManager: Status check:', 
            'API status:', currentModelStatus, 
            'UI status:', currentUIStatus);
        
        // Status has changed from pending to training or completed
        if (currentModelStatus && currentModelStatus !== currentUIStatus) {
            console.log(`ModelDetailManager: Status changed from ${currentUIStatus} to ${currentModelStatus}`);
            
            // Update the data-model-status attribute to reflect actual status
            if (modelElement) {
                modelElement.dataset.modelStatus = currentModelStatus;
                console.log('ModelDetailManager: Updated UI status attribute to', currentModelStatus);
            }
            
            // Only if training has completed/failed, stop updates and reload
            if (!['training', 'pending', 'loading'].includes(currentModelStatus)) {
                console.log(`ModelDetailManager: Training completed with status: ${currentModelStatus}, stopping updates`);
                this.stopUpdates();
                // Reload page after a short delay to show final state
                setTimeout(() => {
                    location.reload();
                }, 2000);
                return;
            }
            
            // If status changed from pending to training, update UI accordingly without reload
            if (currentUIStatus === 'pending' && currentModelStatus === 'training') {
                console.log('ModelDetailManager: Model transitioned from pending to training');
                this.isTraining = true;
                this.showLiveIndicator();
            }
        }
        
        // Update progress bars with latest data
        console.log('ModelDetailManager: Updating progress bars with:', data.progress);
        this.updateProgressBars(data.progress);
        
        // Update metrics
        this.updateMetrics(data.metrics, data.progress);
        
        // Update last update time
        this.updateLastUpdateTime();
        
        // Show success indicator
        this.showLiveIndicator();
        
        // Auto-refresh logs if modal is open
        this.autoRefreshLogs();
        
        // Refresh training preview on epoch completion
        if (data.progress && data.progress.current_epoch) {
            this.refreshTrainingPreview(data.progress.current_epoch);
        }
        
        // Handle training completion (legacy check)
        if (data.status_changed) {
            setTimeout(() => {
                this.refreshTrainingPreview();
                setTimeout(() => {
                    location.reload();
                }, 2000);
            }, 1000);
        }
    }
    
    updateProgressBars(progress) {
        console.log('ModelDetailManager: updateProgressBars called with:', progress);
        console.log('  progressBar element:', !!this.elements.progressBar);
        console.log('  batchProgressBar element:', !!this.elements.batchProgressBar);
        console.log('  batchText element:', !!this.elements.batchText);
        
        if (!progress) {
            console.log('  No progress data provided');
            return;
        }
        
        // Re-query DOM if elements weren't found during initialization
        // This helps with late-rendered elements or page changes
        if (!this.elements.progressBar) {
            this.elements.progressBar = document.querySelector('#training-progress .progress-enhanced .progress-bar');
            console.log('  Re-queried progressBar:', !!this.elements.progressBar);
        }
        if (!this.elements.batchProgressBar) {
            this.elements.batchProgressBar = document.querySelector('#training-progress .progress-sm .progress-bar.bg-info');
            console.log('  Re-queried batchProgressBar:', !!this.elements.batchProgressBar);
        }
        if (!this.elements.batchText) {
            this.elements.batchText = document.querySelector('#training-progress small.text-muted.mb-3.d-block');
            console.log('  Re-queried batchText:', !!this.elements.batchText);
        }
        
        // Show progress bars if model is training
        const isTraining = progress.status && ['training', 'pending', 'loading'].includes(progress.status);
        
        // Main progress bar (epoch progress)
        if (this.elements.progressBar) {
            const progressContainer = this.elements.progressBar.closest('.progress.progress-enhanced');
            if (progressContainer) {
                progressContainer.style.display = isTraining ? 'block' : 'none';
            }
            
            if (isTraining && progress) {
                const percentage = progress.progress_percentage || 0;
                const currentEpoch = progress.current_epoch || 0;
                const totalEpochs = progress.total_epochs || 0;
                
                console.log(`  Updating main progress: ${percentage}% (${currentEpoch}/${totalEpochs})`);
                
                this.elements.progressBar.style.width = percentage + '%';
                this.elements.progressBar.textContent = `Epoch ${currentEpoch}/${totalEpochs} (${percentage.toFixed(1)}%)`;
                this.elements.progressBar.setAttribute('aria-valuenow', currentEpoch);
            }
        } else {
            console.log('  Main progress bar not found in DOM');
        }
        
        // Batch progress bar
        if (this.elements.batchProgressBar) {
            const batchContainer = this.elements.batchProgressBar.closest('.progress.progress-sm');
            if (batchContainer) {
                batchContainer.style.display = (isTraining && progress.batch_progress_percentage !== undefined) ? 'block' : 'none';
            }
            
            if (isTraining && progress.batch_progress_percentage !== undefined) {
                console.log(`  Updating batch progress: ${progress.batch_progress_percentage}% (${progress.current_batch})`);
                this.elements.batchProgressBar.style.width = progress.batch_progress_percentage + '%';
                this.elements.batchProgressBar.setAttribute('aria-valuenow', progress.current_batch);
            }
        } else {
            console.log('  Batch progress bar not found in DOM');
        }
        
        // Batch text
        if (this.elements.batchText) {
            if (isTraining && progress.current_batch && progress.total_batches_per_epoch) {
                const newText = `Batch ${progress.current_batch}/${progress.total_batches_per_epoch} in current epoch`;
                console.log(`  Updating batch text: ${newText}`);
                this.elements.batchText.textContent = newText;
                this.elements.batchText.style.display = 'block';
            } else {
                this.elements.batchText.style.display = 'none';
            }
        } else {
            console.log('  Batch text not found in DOM');
        }
    }
    
    updateMetrics(metrics, progress) {
        if (!metrics) {
            return;
        }
        
        const updateMetric = (id, value, isInteger = false) => {
            const element = document.getElementById(id);
            if (element && value !== null && value !== undefined) {
                const formattedValue = isInteger ? value.toString() : 
                                     (typeof value === 'number' ? value.toFixed(4) : value);
                
                // Only update if value has changed to prevent unnecessary flashing
                if (element.textContent !== formattedValue) {
                    element.textContent = formattedValue;
                    
                    // Add subtle flash effect for updates
                    element.style.backgroundColor = '#e7f3ff';
                    setTimeout(() => {
                        element.style.backgroundColor = '';
                    }, 500);
                }
            }
        };
        
        if (progress) {
            updateMetric('current-epoch', progress.current_epoch, true); // Integer formatting
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
        console.log('ModelDetailManager: manualRefresh() called');
        
        if (this.elements.manualRefreshBtn) {
            this.elements.manualRefreshBtn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Refreshing...';
        }
        
        this.updateProgress();
        
        setTimeout(() => {
            if (this.elements.manualRefreshBtn) {
                this.elements.manualRefreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i>';
            }
        }, 1000);
    }
    
    stopTraining() {
        if (!confirm('Are you sure you want to stop training? This cannot be undone.')) {
            return;
        }
        
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
    
    refreshTrainingPreview(currentEpoch = null) {
        console.log('Refreshing training preview images...');
        
        const trainingSamplesCards = Array.from(document.querySelectorAll('.card-header h5')).filter(h5 => 
            h5.textContent.includes('Training Samples')
        );
        
        if (trainingSamplesCards.length === 0) {
            console.log('No training samples card found');
            return;
        }
        
        const currentTrainingSamplesCard = trainingSamplesCards[0].closest('.card');
        
        fetch(window.location.href, {
            method: 'GET',
            headers: {
                ...setupCSRF(),
                'X-Requested-With': 'XMLHttpRequest'
            },
            credentials: 'same-origin'
        })
        .then(response => response.text())
        .then(html => {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            
            const newTrainingSamplesCards = Array.from(doc.querySelectorAll('.card-header h5')).filter(h5 => 
                h5.textContent.includes('Training Samples')
            );
            
            if (newTrainingSamplesCards.length > 0) {
                const newCard = newTrainingSamplesCards[0].closest('.card');
                const newCardBody = newCard.querySelector('.card-body');
                const currentCardBody = currentTrainingSamplesCard.querySelector('.card-body');
                
                if (newCardBody && currentCardBody) {
                    currentCardBody.innerHTML = newCardBody.innerHTML;
                    console.log('Training preview refreshed successfully');
                    this.showRefreshNotification(currentEpoch);
                }
            }
        })
        .catch(error => {
            console.error('Error refreshing training preview:', error);
            this.showRefreshNotification(currentEpoch, true);
        });
    }
    
    showRefreshNotification(epoch, isError = false) {
        const notification = document.createElement('div');
        notification.className = `alert ${isError ? 'alert-danger' : 'alert-info'} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 300px;';
        
        const icon = isError ? 'fas fa-exclamation-triangle' : 'fas fa-sync-alt';
        const message = isError 
            ? `Error refreshing training images${epoch ? ` for epoch ${epoch}` : ''}`
            : `Training images updated${epoch ? ` for epoch ${epoch}` : ''}`;
        
        notification.innerHTML = `
            <i class="${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    }
    
    showAlert(type, message) {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alert);
        
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
    
    // Logs Modal Functions
    openLogsModal() {
        const modal = new bootstrap.Modal(document.getElementById('logsModal'));
        modal.show();
        
        // Check if we have preloaded logs data
        const logsContent = document.getElementById('modal-logs-content');
        if (this.logsData) {
            // Use the preloaded logs data immediately
            console.log('ModelDetailManager: Using preloaded logs data');
            this.displayLogsData(this.logsData, logsContent);
        } else {
            // Otherwise load logs normally
            this.loadModalLogs();
        }
    }
    
    loadModalLogs(showAll = false) {
        const logsContent = document.getElementById('modal-logs-content');
        if (!logsContent) return;
        
        if (!logsContent.querySelector('.loading-indicator')) {
            logsContent.innerHTML = `
                <div class="text-center py-3 loading-indicator">
                    <i class="fas fa-spinner fa-spin fa-2x text-primary"></i>
                    <p class="mt-2">Loading logs...</p>
                </div>
            `;
        }
        
        const url = `/ml/model/${this.modelId}/logs/`;
        const params = showAll ? '?show_all=true' : '';
        
        fetch(url + params, {
            method: 'GET',
            headers: setupCSRF(),
            credentials: 'same-origin'
        })
            .then(response => response.json())
            .then(data => {
                this.logsData = data; // Store for future use
                this.displayLogsData(data, logsContent);
            })
            .catch(error => {
                console.error('Error loading logs:', error);
                logsContent.innerHTML = '<p class="text-danger">Error loading logs. Please try again.</p>';
            });
    }
    
    displayLogsData(data, logsContent) {
        if (!logsContent) return;
        
        if (data.status === 'success' && data.logs && data.logs.length > 0) {
            // Format structured logs with proper line breaks and styling
            const formattedLogs = data.logs.map(log => {
                const timestamp = log.timestamp || '';
                const level = log.level || 'INFO';
                const content = log.content || '';
                
                // Create colored log line based on level
                let levelClass = 'text-info';
                if (level === 'ERROR') levelClass = 'text-danger';
                else if (level === 'WARNING') levelClass = 'text-warning';
                else if (level === 'DEBUG') levelClass = 'text-muted';
                
                return `<div class="log-line mb-1">
                    <span class="text-muted">${timestamp}</span> 
                    <span class="${levelClass} fw-bold">[${level}]</span> 
                    <span>${this.escapeHtml(content)}</span>
                </div>`;
            }).join('');
            
            logsContent.innerHTML = formattedLogs;
        } else if (typeof data === 'string' && data.trim()) {
            // Fallback for plain text logs
            const formattedLogs = data.split('\n').map(line => 
                line.trim() ? `<div class="log-line">${this.escapeHtml(line)}</div>` : '<br>'
            ).join('');
            
            logsContent.innerHTML = formattedLogs || '<p class="text-muted">No logs available yet.</p>';
        } else {
            logsContent.innerHTML = '<p class="text-muted">No logs available yet.</p>';
        }
        
        // Auto-scroll to bottom
        logsContent.scrollTop = logsContent.scrollHeight;
    }
    
    autoRefreshLogs() {
        // Auto-refresh logs if modal is open and training is active
        const modal = document.getElementById('logsModal');
        if (modal && modal.classList.contains('show') && this.isTraining) {
            this.loadModalLogs();
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Image Navigation System for Training Epochs
class EpochImageNavigator {
    constructor() {
        this.currentEpochIndex = 0;
        this.epochs = [];
        this.init();
    }
    
    init() {
        this.collectEpochs();
        this.addNavigationControls();
    }
    
    collectEpochs() {
        // Collect all epoch images
        const epochElements = document.querySelectorAll('[data-epoch]');
        this.epochs = Array.from(epochElements).map(el => ({
            epoch: parseInt(el.dataset.epoch),
            element: el,
            imageUrl: el.querySelector('img')?.src
        })).sort((a, b) => a.epoch - b.epoch);
        
        console.log(`Found ${this.epochs.length} epoch images`);
    }
    
    addNavigationControls() {
        if (this.epochs.length <= 1) return;
        
        const container = document.querySelector('.training-samples-container, .card-body');
        if (!container) return;
        
        const navControls = document.createElement('div');
        navControls.className = 'epoch-navigation d-flex justify-content-between align-items-center mb-3';
        navControls.innerHTML = `
            <button class="btn btn-outline-primary btn-sm" id="prevEpoch" title="Previous Epoch">
                <i class="fas fa-chevron-left"></i> Previous
            </button>
            <span class="badge bg-primary" id="epochIndicator">
                Epoch ${this.epochs[this.currentEpochIndex]?.epoch || 1} of ${this.epochs.length}
            </span>
            <button class="btn btn-outline-primary btn-sm" id="nextEpoch" title="Next Epoch">
                Next <i class="fas fa-chevron-right"></i>
            </button>
        `;
        
        container.insertBefore(navControls, container.firstChild);
        
        // Add event listeners
        document.getElementById('prevEpoch').addEventListener('click', () => this.previousEpoch());
        document.getElementById('nextEpoch').addEventListener('click', () => this.nextEpoch());
        
        // Initialize display
        this.updateDisplay();
    }
    
    previousEpoch() {
        if (this.currentEpochIndex > 0) {
            this.currentEpochIndex--;
            this.updateDisplay();
        }
    }
    
    nextEpoch() {
        if (this.currentEpochIndex < this.epochs.length - 1) {
            this.currentEpochIndex++;
            this.updateDisplay();
        }
    }
    
    updateDisplay() {
        // Hide all epoch images
        this.epochs.forEach((epoch, index) => {
            epoch.element.style.display = index === this.currentEpochIndex ? 'block' : 'none';
        });
        
        // Update navigation controls
        const prevBtn = document.getElementById('prevEpoch');
        const nextBtn = document.getElementById('nextEpoch');
        const indicator = document.getElementById('epochIndicator');
        
        if (prevBtn) prevBtn.disabled = this.currentEpochIndex === 0;
        if (nextBtn) nextBtn.disabled = this.currentEpochIndex === this.epochs.length - 1;
        
        if (indicator) {
            const currentEpoch = this.epochs[this.currentEpochIndex];
            indicator.textContent = `Epoch ${currentEpoch?.epoch || 1} of ${this.epochs.length}`;
        }
    }
    
    // Auto-advance to latest epoch when new images arrive
    autoAdvanceToLatest() {
        const newEpochCount = document.querySelectorAll('[data-epoch]').length;
        if (newEpochCount > this.epochs.length) {
            this.collectEpochs();
            this.currentEpochIndex = this.epochs.length - 1; // Go to latest
            this.updateDisplay();
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Get model ID from data attribute
    const modelElement = document.querySelector('[data-model-id]');
    if (!modelElement) {
        console.warn('Model ID not found, skipping model detail manager initialization');
        return;
    }
    
    const modelId = modelElement.dataset.modelId;
    console.log('Initializing ModelDetailManager for model:', modelId);
    
    // Initialize unified manager
    window.modelDetailManager = new ModelDetailManager(modelId);
    
    // Initialize epoch navigation if training images exist
    const hasTrainingImages = document.querySelectorAll('[data-epoch]').length > 0;
    if (hasTrainingImages) {
        window.epochNavigator = new EpochImageNavigator();
        
        // Auto-refresh epoch navigation when images update
        if (window.modelDetailManager.isTraining) {
            setInterval(() => {
                if (window.epochNavigator) {
                    window.epochNavigator.autoAdvanceToLatest();
                }
            }, 5000);
        }
    }
});

// Add CSS for smooth animations
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .epoch-navigation {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    .epoch-navigation .btn {
        min-width: 80px;
    }
    
    .training-sample-image {
        transition: opacity 0.3s ease;
    }
    
    .metric-value {
        transition: background-color 0.5s ease;
    }
    
    .log-line {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.4;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    
    .log-line:hover {
        background-color: #f8f9fa;
    }
`;
document.head.appendChild(style);
