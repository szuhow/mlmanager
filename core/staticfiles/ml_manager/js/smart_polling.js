/**
 * Smart Polling Manager - Efficient alternative to constant GET polling
 * Features:
 * - ETag-based conditional requests (304 Not Modified responses)
 * - Exponential backoff when no changes detected
 * - Automatic stop when training completes
 * - Memory of last update time to reduce unnecessary requests
 * - Connection error handling with backoff
 * - Real-time log streaming support
 */

class SmartPollingManager {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || window.location.href;
        this.logsUrl = options.logsUrl || `${this.baseUrl}logs/realtime/`;
        this.minInterval = options.minInterval || 2000; // 2 seconds minimum
        this.maxInterval = options.maxInterval || 15000; // 15 seconds maximum for logs
        this.backoffMultiplier = options.backoffMultiplier || 1.3;
        this.resetOnChange = options.resetOnChange !== false; // Reset interval on changes
        
        // State management
        this.currentInterval = this.minInterval;
        this.isActive = false;
        this.timeoutId = null;
        this.etag = null;
        this.lastModified = null;
        this.lastUpdateTime = null;
        this.lastLogTimestamp = null;
        this.consecutiveNoChanges = 0;
        this.errorCount = 0;
        this.maxErrors = 3;
        
        // Callbacks
        this.onUpdate = options.onUpdate || (() => {});
        this.onLogsUpdate = options.onLogsUpdate || (() => {});
        this.onError = options.onError || (() => {});
        this.onStatusChange = options.onStatusChange || (() => {});
        
        console.log('SmartPollingManager initialized with options:', options);
    }

    /**
     * Start polling for training data and logs
     */
    start() {
        if (this.isActive) {
            console.log('SmartPollingManager already active');
            return;
        }
        
        this.isActive = true;
        this.currentInterval = this.minInterval;
        this.consecutiveNoChanges = 0;
        this.errorCount = 0;
        
        console.log('SmartPollingManager started with logs support');
        this.onStatusChange({ status: 'active', interval: this.currentInterval });
        
        // Start both data and logs polling
        this._scheduleNextPoll();
        this._pollLogs(); // Start logs polling immediately
    }

    /**
     * Stop all polling
     */
    stop() {
        if (!this.isActive) return;
        
        this.isActive = false;
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
        
        console.log('SmartPollingManager stopped');
        this.onStatusChange({ status: 'stopped' });
    }

    /**
     * Schedule next training data poll
     */
    _scheduleNextPoll() {
        if (!this.isActive) return;
        
        this.timeoutId = setTimeout(() => {
            this._poll();
        }, this.currentInterval);
        
        console.log(`Next poll scheduled in ${this.currentInterval}ms`);
    }

    /**
     * Poll for real-time logs with efficient timestamp-based filtering
     */
    async _pollLogs() {
        if (!this.isActive) return;
        
        try {
            const url = new URL(this.logsUrl);
            if (this.lastLogTimestamp) {
                url.searchParams.set('since', this.lastLogTimestamp);
            }

            const response = await fetch(url, {
                method: 'GET',
                credentials: 'same-origin',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (response.ok) {
                const logsData = await response.json();
                
                if (logsData.logs && logsData.logs.length > 0) {
                    // Update last timestamp for next request
                    const timestamps = logsData.logs.map(log => log.timestamp).filter(t => t);
                    if (timestamps.length > 0) {
                        this.lastLogTimestamp = Math.max(...timestamps.map(t => new Date(t).getTime()));
                    }
                    
                    this.onLogsUpdate(logsData);
                    console.log(`Received ${logsData.logs.length} new log entries`);
                }
            }

        } catch (error) {
            console.error('SmartPolling: Logs fetch error:', error);
            this.onError(error);
        } finally {
            // Schedule next logs poll with shorter interval (logs are more time-sensitive)
            if (this.isActive) {
                setTimeout(() => this._pollLogs(), Math.min(this.currentInterval, 3000));
            }
        }
    }

    /**
     * Main polling function for training data
     */
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
                
                // Process the update
                this._handleUpdate(data);
                this.errorCount = 0; // Reset error count on success
                
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
        } catch (error) {
            console.error('SmartPolling: Poll error:', error);
            this._handleError(error);
        } finally {
            if (this.isActive) {
                this._scheduleNextPoll();
            }
        }
    }

    /**
     * Handle successful data update
     */
    _handleUpdate(data) {
        this.lastUpdateTime = Date.now();
        this.consecutiveNoChanges = 0;
        
        // Reset interval on significant changes
        if (this.resetOnChange && this._hasSignificantChanges(data)) {
            this.currentInterval = this.minInterval;
            console.log('Significant changes detected, resetting interval to minimum');
        }
        
        this.onUpdate(data);
        
        // Check if training is complete and stop polling if needed
        if (data.status && !this._isTrainingStatus(data.status)) {
            console.log('Training completed, stopping polling');
            this.stop();
        }
    }

    /**
     * Handle no changes detected
     */
    _handleNoChanges() {
        this.consecutiveNoChanges++;
        
        // Gradually increase interval for efficiency
        if (this.consecutiveNoChanges > 2) {
            this.currentInterval = Math.min(
                this.currentInterval * this.backoffMultiplier,
                this.maxInterval
            );
            console.log(`No changes for ${this.consecutiveNoChanges} polls, increasing interval to ${this.currentInterval}ms`);
        }
    }

    /**
     * Handle errors with exponential backoff
     */
    _handleError(error) {
        this.errorCount++;
        
        if (this.errorCount >= this.maxErrors) {
            console.error('Too many consecutive errors, stopping polling');
            this.stop();
            this.onError({ 
                type: 'max_errors_reached', 
                error: error, 
                count: this.errorCount 
            });
            return;
        }
        
        // Increase interval on errors
        this.currentInterval = Math.min(
            this.currentInterval * 2,
            this.maxInterval
        );
        
        console.warn(`Error ${this.errorCount}/${this.maxErrors}, increasing interval to ${this.currentInterval}ms`);
        this.onError({ 
            type: 'fetch_error', 
            error: error, 
            count: this.errorCount 
        });
    }

    /**
     * Check if status indicates active training
     */
    _isTrainingStatus(status) {
        return ['PENDING', 'STARTED', 'PROGRESS'].includes(status);
    }

    /**
     * Determine if changes are significant enough to reset polling interval
     */
    _hasSignificantChanges(data) {
        // Consider status changes or progress updates as significant
        return data.status_changed || 
               (data.progress && data.progress > 0) ||
               data.new_metrics || 
               data.logs_updated;
    }

    /**
     * Force an immediate poll (useful for user-triggered updates)
     */
    forcePoll() {
        if (!this.isActive) {
            console.log('Cannot force poll - manager is not active');
            return;
        }
        
        console.log('Forcing immediate poll');
        clearTimeout(this.timeoutId);
        this._poll();
    }

    /**
     * Reset polling interval to minimum (useful after user interactions)
     */
    resetInterval() {
        this.currentInterval = this.minInterval;
        this.consecutiveNoChanges = 0;
        console.log('Polling interval reset to minimum');
    }

    /**
     * Get current polling status
     */
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

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SmartPollingManager;
}