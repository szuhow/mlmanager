/**
 * Smart Polling Manager - Efficient alternative to constant GET polling
 * Features:
 * - ETag-based conditional requests (304 Not Modified responses)
 * - Exponential backoff when no changes detected
 * - Automatic stop when training completes
 * - Memory of last update time to reduce unnecessary requests
 * - Connection error handling with backoff
 */

class SmartPollingManager {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || window.location.href;
        this.minInterval = options.minInterval || 2000; // 2 seconds minimum
        this.maxInterval = options.maxInterval || 30000; // 30 seconds maximum
        this.backoffMultiplier = options.backoffMultiplier || 1.5;
        this.resetOnChange = options.resetOnChange !== false; // Reset interval on changes
        
        // State management
        this.currentInterval = this.minInterval;
        this.isActive = false;
        this.timeoutId = null;
        this.etag = null;
        this.lastModified = null;
        this.lastUpdateTime = null;
        this.consecutiveNoChanges = 0;
        this.errorCount = 0;
        this.maxErrors = 3;
        
        // Callbacks
        this.onUpdate = options.onUpdate || (() => {});
        this.onError = options.onError || (() => {});
        this.onStatusChange = options.onStatusChange || (() => {});
        
        console.log('SmartPollingManager initialized with options:', options);
    }
    
    start() {
        if (this.isActive) {
            console.log('SmartPollingManager already active');
            return;
        }
        
        this.isActive = true;
        this.currentInterval = this.minInterval;
        this.consecutiveNoChanges = 0;
        this.errorCount = 0;
        
        console.log('SmartPollingManager started');
        this._scheduleNextPoll();
    }
    
    stop() {
        if (!this.isActive) return;
        
        this.isActive = false;
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
        
        console.log('SmartPollingManager stopped');
    }
    
    _scheduleNextPoll() {
        if (!this.isActive) return;
        
        this.timeoutId = setTimeout(() => {
            this._poll();
        }, this.currentInterval);
        
        console.log(`Next poll scheduled in ${this.currentInterval}ms`);
    }
    
    async _poll() {
        if (!this.isActive) return;
        
        try {
            const headers = {
                'X-Requested-With': 'XMLHttpRequest'
            };
            
            // Add conditional request headers for efficient polling
            if (this.etag) {
                headers['If-None-Match'] = this.etag;
            }
            if (this.lastModified) {
                headers['If-Modified-Since'] = this.lastModified;
            }
            
            const response = await fetch(this.baseUrl, { headers });
            
            // Handle 304 Not Modified - no changes
            if (response.status === 304) {
                console.log('No changes detected (304 Not Modified)');
                this._handleNoChanges();
                this._scheduleNextPoll();
                return;
            }
            
            // Handle successful response with potential changes
            if (response.ok) {
                const data = await response.json();
                
                // Update cache headers
                this.etag = response.headers.get('ETag');
                this.lastModified = response.headers.get('Last-Modified');
                this.lastUpdateTime = Date.now();
                
                // Reset error count on success
                this.errorCount = 0;
                
                // Check if training is complete
                if (data.status && !this._isTrainingStatus(data.status)) {
                    console.log(`Training completed with status: ${data.status}`);
                    this.onStatusChange(data.status);
                    this.stop();
                    return;
                }
                
                // Determine if there were actual changes worth reporting
                const hasChanges = this._hasSignificantChanges(data);
                
                if (hasChanges) {
                    console.log('Significant changes detected, updating UI');
                    this.onUpdate(data);
                    
                    if (this.resetOnChange) {
                        this.currentInterval = this.minInterval;
                        this.consecutiveNoChanges = 0;
                    }
                } else {
                    console.log('Response received but no significant changes');
                    this._handleNoChanges();
                }
                
                this._scheduleNextPoll();
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
        } catch (error) {
            console.error('Polling error:', error);
            this._handleError(error);
        }
    }
    
    _handleNoChanges() {
        this.consecutiveNoChanges++;
        
        // Increase polling interval using exponential backoff
        if (this.consecutiveNoChanges > 2) { // Start backing off after 3 no-change responses
            this.currentInterval = Math.min(
                this.currentInterval * this.backoffMultiplier,
                this.maxInterval
            );
            console.log(`Backing off to ${this.currentInterval}ms due to ${this.consecutiveNoChanges} consecutive no-changes`);
        }
    }
    
    _handleError(error) {
        this.errorCount++;
        this.onError(error);
        
        if (this.errorCount >= this.maxErrors) {
            console.error(`Too many consecutive errors (${this.errorCount}), stopping polling`);
            this.stop();
            return;
        }
        
        // Exponential backoff for errors
        const errorBackoffInterval = Math.min(
            this.currentInterval * Math.pow(2, this.errorCount),
            this.maxInterval
        );
        
        console.log(`Error ${this.errorCount}/${this.maxErrors}, retrying in ${errorBackoffInterval}ms`);
        
        if (this.isActive) {
            this.timeoutId = setTimeout(() => {
                this._poll();
            }, errorBackoffInterval);
        }
    }
    
    _isTrainingStatus(status) {
        return ['training', 'loading', 'pending'].includes(status.toLowerCase());
    }
    
    _hasSignificantChanges(data) {
        // Define what constitutes significant changes worth updating the UI
        // This can be customized based on your needs
        
        if (!this._lastData) {
            this._lastData = data;
            return true; // First response is always significant
        }
        
        const significant = (
            // Status changes
            data.status !== this._lastData.status ||
            
            // Progress changes (epoch advancement)
            (data.progress && this._lastData.progress && 
             data.progress.current_epoch !== this._lastData.progress.current_epoch) ||
            
            // New log entries
            (data.logs && this._lastData.logs &&
             data.logs.length !== this._lastData.logs.length) ||
            
            // Metric improvements
            (data.progress && this._lastData.progress &&
             data.progress.best_val_dice !== this._lastData.progress.best_val_dice)
        );
        
        this._lastData = data;
        return significant;
    }
    
    // Force immediate poll (useful for manual refresh)
    forcePoll() {
        if (!this.isActive) {
            console.log('Cannot force poll - SmartPollingManager not active');
            return;
        }
        
        console.log('Force polling requested');
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
        }
        this._poll();
    }
    
    // Reset polling interval to minimum (useful when user interaction indicates they want fresh data)
    resetInterval() {
        this.currentInterval = this.minInterval;
        this.consecutiveNoChanges = 0;
        console.log('Polling interval reset to minimum');
    }
    
    getStatus() {
        return {
            isActive: this.isActive,
            currentInterval: this.currentInterval,
            consecutiveNoChanges: this.consecutiveNoChanges,
            errorCount: this.errorCount,
            lastUpdateTime: this.lastUpdateTime
        };
    }
}

// Alternative: Long Polling implementation
class LongPollingManager {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || window.location.href;
        this.timeout = options.timeout || 30000; // 30 second timeout
        this.isActive = false;
        this.controller = null;
        
        this.onUpdate = options.onUpdate || (() => {});
        this.onError = options.onError || (() => {});
        this.onStatusChange = options.onStatusChange || (() => {});
        
        console.log('LongPollingManager initialized');
    }
    
    start() {
        if (this.isActive) return;
        
        this.isActive = true;
        console.log('LongPollingManager started');
        this._startLongPoll();
    }
    
    stop() {
        this.isActive = false;
        if (this.controller) {
            this.controller.abort();
            this.controller = null;
        }
        console.log('LongPollingManager stopped');
    }
    
    async _startLongPoll() {
        while (this.isActive) {
            try {
                this.controller = new AbortController();
                
                const response = await fetch(this.baseUrl + '?long_poll=true', {
                    method: 'GET',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    },
                    signal: this.controller.signal
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.status && !this._isTrainingStatus(data.status)) {
                        this.onStatusChange(data.status);
                        this.stop();
                        return;
                    }
                    
                    this.onUpdate(data);
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
                
            } catch (error) {
                if (error.name === 'AbortError') {
                    console.log('Long poll aborted');
                    break;
                }
                
                console.error('Long polling error:', error);
                this.onError(error);
                
                // Brief pause before retry
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
        }
    }
    
    _isTrainingStatus(status) {
        return ['training', 'loading', 'pending'].includes(status.toLowerCase());
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SmartPollingManager, LongPollingManager };
} else {
    window.SmartPollingManager = SmartPollingManager;
    window.LongPollingManager = LongPollingManager;
}
