// Real-time progress updates for ML models
class ModelProgressUpdater {
    constructor() {
        this.updateInterval = 5000; // 5 seconds
        this.fastUpdateInterval = 2000; // 2 seconds for pending models
        this.intervalId = null;
        this.isUpdating = false;
        this.pendingModels = new Set(); // Track pending models
        this.currentInterval = this.updateInterval;
    }

    start() {
        if (this.isUpdating) return;
        
        this.isUpdating = true;
        this.updateProgress();
        this.intervalId = setInterval(() => this.updateProgress(), this.currentInterval);
        
        // Add visual indicator
        this.showUpdateIndicator();
    }

    restart() {
        this.stop();
        setTimeout(() => this.start(), 100);
    }

    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.isUpdating = false;
        this.hideUpdateIndicator();
    }

    async updateProgress() {
        console.log('ModelProgressUpdater: updateProgress called');
        try {
            // Find both training and pending models
            const trainingRows = document.querySelectorAll('tr[data-model-status="training"]:not([data-model-deleted="true"])');
            const pendingRows = document.querySelectorAll('tr[data-model-status="pending"]:not([data-model-deleted="true"])');
            
            console.log('ModelProgressUpdater: Found', trainingRows.length, 'training rows and', pendingRows.length, 'pending rows to update');
            
            // Update pending models set
            this.pendingModels.clear();
            pendingRows.forEach(row => {
                const modelId = row.dataset.modelId;
                if (modelId) this.pendingModels.add(modelId);
            });
            
            // Adjust update interval based on pending models
            const newInterval = this.pendingModels.size > 0 ? this.fastUpdateInterval : this.updateInterval;
            if (newInterval !== this.currentInterval) {
                console.log('ModelProgressUpdater: Changing update interval from', this.currentInterval, 'to', newInterval);
                this.currentInterval = newInterval;
                this.restart();
                return;
            }
            
            if (trainingRows.length === 0 && pendingRows.length === 0) {
                console.log('ModelProgressUpdater: No training or pending models, stopping updates');
                this.stop();
                return;
            }

            // Update all models
            for (const row of [...trainingRows, ...pendingRows]) {
                const modelId = row.dataset.modelId;
                console.log('ModelProgressUpdater: Updating model', modelId);
                if (!modelId) continue;

                await this.updateModelRow(modelId, row);
            }
        } catch (error) {
            console.error('ModelProgressUpdater: Error updating progress:', error);
        }
    }

    // Methods to handle dropdown conflicts
    pauseUpdateForRow(row) {
        if (row) {
            row.setAttribute('data-update-paused', 'true');
        }
    }

    resumeUpdateForRow(row) {
        if (row) {
            row.removeAttribute('data-update-paused');
        }
    }

    // Enhanced update method that respects paused rows
    async updateModelRow(modelId, row) {
        // Skip update if row is paused (dropdown open)
        if (row && row.getAttribute('data-update-paused') === 'true') {
            return;
        }

        try {
            const response = await fetch(`/ml/model/${modelId}/progress/`, {
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (!response.ok) {
                // Handle 404 errors - model was deleted
                if (response.status === 404) {
                    console.warn(`Model ${modelId} not found (404) - removing from live updates`);
                    this.removeModelFromUpdates(modelId, row);
                    return;
                }
                console.warn(`Failed to fetch model ${modelId}: ${response.status}`);
                return;
            }

            const data = await response.json();
            
            // Skip DOM updates if row is paused
            if (row && row.getAttribute('data-update-paused') === 'true') {
                return;
            }
            
            // Update progress bar
            const progressBar = row.querySelector('.progress-bar');
            if (progressBar && data.progress) {
                const progressPercent = data.progress.percentage || 0;
                progressBar.style.width = `${progressPercent}%`;
                progressBar.textContent = `${data.progress.current_epoch}/${data.progress.total_epochs}`;
            }

            // Update performance
            const performanceCell = row.querySelector('.performance-cell');
            if (performanceCell && data.metrics && data.metrics.best_val_dice > 0) {
                performanceCell.innerHTML = `<span class="badge bg-info">${data.metrics.best_val_dice.toFixed(3)}</span>`;
            }

            // Update status if changed
            if (data.model_status && data.model_status !== row.dataset.modelStatus) {
                console.log('ModelProgressUpdater: Status changed from', row.dataset.modelStatus, 'to', data.model_status, 'for model', modelId);
                
                // If model was pending and now started training, show notification
                if (row.dataset.modelStatus === 'pending' && data.model_status === 'training') {
                    this.showStatusChangeNotification(modelId, 'Training Started', 'Model has started training successfully');
                }
                
                // Status changed, reload page to reflect new state
                setTimeout(() => {
                    window.location.reload();
                }, 1000); // Small delay to show the notification
            }

        } catch (error) {
            console.error(`Error updating model ${modelId}:`, error);
        }
    }

    removeModelFromUpdates(modelId, row) {
        // Add visual indication that model was deleted
        if (row) {
            row.style.opacity = '0.5';
            row.style.backgroundColor = '#f8d7da';
            
            // Add a notice to the row
            const nameCell = row.querySelector('td:nth-child(2)');
            if (nameCell) {
                const notice = document.createElement('small');
                notice.className = 'text-danger d-block';
                notice.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Model deleted';
                nameCell.appendChild(notice);
            }
            
            // Remove checkbox to prevent selection
            const checkbox = row.querySelector('.model-checkbox');
            if (checkbox) {
                checkbox.disabled = true;
                checkbox.checked = false;
            }
        }
        
        // Mark row as deleted to skip future updates
        if (row) {
            row.setAttribute('data-model-deleted', 'true');
        }
    }

    showUpdateIndicator() {
        let indicator = document.getElementById('update-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'update-indicator';
            indicator.className = 'alert alert-info alert-dismissible fade show position-fixed';
            indicator.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
            
            const intervalText = this.currentInterval === this.fastUpdateInterval ? 
                `${this.currentInterval/1000}s (fast mode for pending models)` : 
                `${this.currentInterval/1000}s`;
                
            indicator.innerHTML = `
                <i class="fas fa-sync-alt fa-spin"></i>
                <strong>Live Updates Active</strong>
                <small class="d-block">Progress updates every ${intervalText}</small>
                <button type="button" class="btn-close" onclick="modelProgressUpdater.stop()"></button>
            `;
            document.body.appendChild(indicator);
        } else {
            // Update existing indicator
            const intervalText = this.currentInterval === this.fastUpdateInterval ? 
                `${this.currentInterval/1000}s (fast mode for pending models)` : 
                `${this.currentInterval/1000}s`;
            const small = indicator.querySelector('small');
            if (small) {
                small.textContent = `Progress updates every ${intervalText}`;
            }
        }
    }

    showStatusChangeNotification(modelId, title, message) {
        const notification = document.createElement('div');
        notification.className = 'alert alert-success alert-dismissible fade show position-fixed';
        notification.style.cssText = 'top: 80px; right: 20px; z-index: 1055; min-width: 350px; max-width: 400px;';
        notification.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-play-circle text-success me-2 fa-lg"></i>
                <div>
                    <strong>${title}</strong>
                    <small class="d-block text-muted">${message}</small>
                </div>
            </div>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    hideUpdateIndicator() {
        const indicator = document.getElementById('update-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
}

// Initialize when page loads
let modelProgressUpdater;

document.addEventListener('DOMContentLoaded', function() {
    console.log('ModelProgressUpdater: DOM loaded, initializing...');
    modelProgressUpdater = new ModelProgressUpdater();
    
    // Auto-start if there are training or pending models
    const trainingModels = document.querySelectorAll('tr[data-model-status="training"]');
    const pendingModels = document.querySelectorAll('tr[data-model-status="pending"]');
    const totalActiveModels = trainingModels.length + pendingModels.length;
    
    console.log('ModelProgressUpdater: Found', trainingModels.length, 'training models and', pendingModels.length, 'pending models');
    if (totalActiveModels > 0) {
        console.log('ModelProgressUpdater: Starting auto-updates for', totalActiveModels, 'active models');
        modelProgressUpdater.start();
    } else {
        console.log('ModelProgressUpdater: No active models found, not starting auto-updates');
    }

    // Add manual refresh button
    const refreshBtn = document.getElementById('manual-refresh');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            modelProgressUpdater.updateProgress();
        });
    }

    // Add toggle button for auto-updates
    const toggleBtn = document.getElementById('toggle-updates');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
            if (modelProgressUpdater.isUpdating) {
                modelProgressUpdater.stop();
                this.innerHTML = '<i class="fas fa-play"></i> Start Live Updates';
                this.className = 'btn btn-outline-success btn-sm';
            } else {
                modelProgressUpdater.start();
                this.innerHTML = '<i class="fas fa-pause"></i> Stop Live Updates';
                this.className = 'btn btn-outline-warning btn-sm';
            }
        });
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (modelProgressUpdater) {
        modelProgressUpdater.stop();
    }
});
