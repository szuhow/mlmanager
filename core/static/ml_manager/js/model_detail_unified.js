/**
 * Unified Model Detail UI Manager
 * Consolidates all update mechanisms to prevent flickering and conflicts
 */

class ModelDetailManager {
    constructor(modelId) {
        this.modelId = modelId;
        this.updateInterval = null;
        this.isTraining = false;
        this.lastUpdateTime = null;
        this.updateIntervalMs = 2000; // 2 seconds
        
        // UI elements cache
        this.elements = {
            updateStatus: document.getElementById('update-status'),
            lastUpdateTime: document.getElementById('last-update-time'),
            progressBar: document.querySelector('.progress-bar'),
            batchProgressBar: document.querySelector('.progress-bar.bg-info'),
            batchText: document.querySelector('small.text-muted.mb-3.d-block'),
            stopBtn: document.getElementById('stopTrainingBtn'),
            manualRefreshBtn: document.getElementById('manualRefreshBtn')
        };
        
        this.init();
    }
    
    init() {
        // Check if model is training
        const modelStatus = document.querySelector('[data-model-status]');
        this.isTraining = modelStatus && modelStatus.dataset.modelStatus === 'training';
        
        if (this.isTraining) {
            this.startUpdates();
            this.showLiveIndicator();
        }
        
        this.setupEventListeners();
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
        // Update progress bars
        this.updateProgressBars(data.progress);
        
        // Update metrics
        this.updateMetrics(data.metrics, data.progress);
        
        // Update last update time
        this.updateLastUpdateTime();
        
        // Show success indicator
        this.showLiveIndicator();
        
        // Refresh training preview on epoch completion
        if (data.progress && data.progress.current_epoch) {
            this.refreshTrainingPreview(data.progress.current_epoch);
        }
        
        // Handle training completion
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
        if (!progress) return;
        
        // Main progress bar (epoch progress)
        if (this.elements.progressBar && progress) {
            this.elements.progressBar.style.width = progress.percentage + '%';
            this.elements.progressBar.textContent = `Epoch ${progress.current_epoch}/${progress.total_epochs} (${progress.percentage.toFixed(1)}%)`;
            this.elements.progressBar.setAttribute('aria-valuenow', progress.current_epoch);
        }
        
        // Batch progress bar
        if (this.elements.batchProgressBar && progress.batch_progress_percentage !== undefined) {
            this.elements.batchProgressBar.style.width = progress.batch_progress_percentage + '%';
            this.elements.batchProgressBar.setAttribute('aria-valuenow', progress.current_batch);
        }
        
        // Batch text
        if (this.elements.batchText && progress.current_batch && progress.total_batches_per_epoch) {
            this.elements.batchText.textContent = `Batch ${progress.current_batch}/${progress.total_batches_per_epoch} in current epoch`;
        }
    }
    
    updateMetrics(metrics, progress) {
        if (!metrics) return;
        
        const updateMetric = (id, value) => {
            const element = document.getElementById(id);
            if (element && value !== null && value !== undefined) {
                const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                
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
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
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
`;
document.head.appendChild(style);
