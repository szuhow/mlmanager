// Real-time progress updates for ML models
class ModelProgressUpdater {
    constructor() {
        this.updateInterval = 5000; // 5 seconds
        this.intervalId = null;
        this.isUpdating = false;
    }

    start() {
        if (this.isUpdating) return;
        
        this.isUpdating = true;
        this.updateProgress();
        this.intervalId = setInterval(() => this.updateProgress(), this.updateInterval);
        
        // Add visual indicator
        this.showUpdateIndicator();
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
            const trainingRows = document.querySelectorAll('tr[data-model-status="training"]:not([data-model-deleted="true"])');
            console.log('ModelProgressUpdater: Found', trainingRows.length, 'training rows to update');
            
            if (trainingRows.length === 0) {
                console.log('ModelProgressUpdater: No training models, stopping updates');
                // No training models, stop updating
                this.stop();
                return;
            }

            for (const row of trainingRows) {
                const modelId = row.dataset.modelId;
                console.log('ModelProgressUpdater: Updating model', modelId);
                if (!modelId) continue;

                await this.updateModelRow(modelId, row);
            }
        } catch (error) {
            console.error('ModelProgressUpdater: Error updating progress:', error);
        }
    }

    async updateModelRow(modelId, row) {
        try {
            const response = await fetch(`/ml/model/${modelId}/`, {
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
            
            // Update progress bar
            const progressBar = row.querySelector('.progress-bar');
            if (progressBar && data.progress) {
                const progressPercent = data.progress.progress_percentage;
                progressBar.style.width = `${progressPercent}%`;
                progressBar.textContent = `${data.progress.current_epoch}/${data.progress.total_epochs}`;
            }

            // Update performance
            const performanceCell = row.querySelector('.performance-cell');
            if (performanceCell && data.progress.best_val_dice > 0) {
                performanceCell.innerHTML = `<span class="badge bg-info">${data.progress.best_val_dice.toFixed(3)}</span>`;
            }

            // Update status if changed
            if (data.status !== 'training') {
                // Status changed, reload page to reflect new state
                window.location.reload();
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
            indicator.innerHTML = `
                <i class="fas fa-sync-alt fa-spin"></i>
                <strong>Live Updates Active</strong>
                <small class="d-block">Progress updates every ${this.updateInterval/1000} seconds</small>
                <button type="button" class="btn-close" onclick="modelProgressUpdater.stop()"></button>
            `;
            document.body.appendChild(indicator);
        }
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
    
    // Auto-start if there are training models
    const trainingModels = document.querySelectorAll('tr[data-model-status="training"]');
    console.log('ModelProgressUpdater: Found', trainingModels.length, 'training models');
    if (trainingModels.length > 0) {
        console.log('ModelProgressUpdater: Starting auto-updates');
        modelProgressUpdater.start();
    } else {
        console.log('ModelProgressUpdater: No training models found, not starting auto-updates');
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
