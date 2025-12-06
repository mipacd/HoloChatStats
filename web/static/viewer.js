// ==UserScript==
// @name         HoloChatStats Live Chat Tracker
// @namespace    http://holochatstats.info/
// @version      1.0
// @description  Track YouTube live chat messages for HoloChatStats
// @author       HoloChatStats
// @match        https://www.youtube.com/live_chat*
// @grant        none
// @run-at       document-idle
// @license      MIT
// ==/UserScript==

(function() {
    'use strict';

    console.log('HoloChatStats Live Chat Tracker initialized');

    let parentOrigin = null;
    let observing = false;
    let processedMessages = new Set();

    // Determine parent origin
    function determineParentOrigin() {
        const urlParams = new URLSearchParams(window.location.search);
        const embedDomain = urlParams.get('embed_domain');
        
        if (!embedDomain) return null;
        
        if (embedDomain.includes('localhost')) {
            parentOrigin = embedDomain.includes(':') ? `http://${embedDomain}` : `http://${embedDomain}:5000`;
        } else {
            parentOrigin = embedDomain.match(/^https?:\/\//) ? embedDomain : `https://${embedDomain}`;
        }
        
        console.log('Parent origin:', parentOrigin);
        return parentOrigin;
    }

    parentOrigin = determineParentOrigin();

    // Notify parent that script is ready
    function notifyParent() {
        if (parentOrigin && window.parent !== window) {
            window.parent.postMessage({ type: 'HOLOCHATSTATS_SCRIPT_READY' }, parentOrigin);
        }
    }

    // Check if element is a valid chat message
    function isValidChatMessage(element) {
        if (!element || !element.tagName) return false;
        
        const tagName = element.tagName.toUpperCase();
        const validTags = [
            'YT-LIVE-CHAT-TEXT-MESSAGE-RENDERER',
            'YT-LIVE-CHAT-PAID-MESSAGE-RENDERER',
            'YT-LIVE-CHAT-MEMBERSHIP-ITEM-RENDERER',
            'YTD-SPONSORSHIPS-LIVE-CHAT-GIFT-REDEMPTION-ANNOUNCEMENT-RENDERER',
            'YT-LIVE-CHAT-PAID-STICKER-RENDERER'
        ];
        
        if (!validTags.includes(tagName)) return false;
        
        const hasAuthor = element.querySelector('#author-name');
        const hasMessage = element.querySelector('#message');
        const hasPurchaseAmount = element.querySelector('#purchase-amount');
        
        return hasAuthor || hasMessage || hasPurchaseAmount;
    }

    // Extract and send message data
    function processMessage(element) {
        if (!isValidChatMessage(element)) return;
        
        const tagName = element.tagName.toUpperCase();
        const authorEl = element.querySelector('#author-name');
        const messageEl = element.querySelector('#message');
        const timestampEl = element.querySelector('#timestamp');
        
        const author = authorEl?.textContent?.trim() || '';
        const message = messageEl?.textContent?.trim() || '';
        const timestamp = timestampEl?.textContent?.trim() || '';
        
        // Create content key for duplicate detection
        const contentKey = `${author}_${message}_${timestamp}_${tagName}`;
        
        if (processedMessages.has(contentKey)) return;
        processedMessages.add(contentKey);

        try {
            const badges = Array.from(element.querySelectorAll('[aria-label*="Member"], [aria-label*="member"], [aria-label*="Sponsor"], [aria-label*="sponsor"]'));
            
            const data = {
                id: contentKey,
                author: author || 'Unknown',
                message: message,
                isMember: badges.length > 0,
                badges: badges.map(b => b.getAttribute('aria-label')),
                rawTimestamp: Date.now()
            };

            // Check for SuperChat
            if (tagName === 'YT-LIVE-CHAT-PAID-MESSAGE-RENDERER' || 
                tagName === 'YT-LIVE-CHAT-PAID-STICKER-RENDERER') {
                data.isSuperChat = true;
                const amountEl = element.querySelector('#purchase-amount');
                data.superChatAmount = amountEl?.textContent?.trim() || '';
            }

            // Check for membership gift redemptions
            if (tagName === 'YTD-SPONSORSHIPS-LIVE-CHAT-GIFT-REDEMPTION-ANNOUNCEMENT-RENDERER') {
                data.isMembershipGiftRedemption = true;
            }

            // Check for new memberships / membership purchases
            if (tagName === 'YT-LIVE-CHAT-MEMBERSHIP-ITEM-RENDERER') {
                data.isNewMember = true;
                if (element.hasAttribute('show-only-header')) {
                    data.isMembershipPurchase = true;
                }
            }

            if (data.author !== 'Unknown' || data.message || data.isSuperChat) {
                if (parentOrigin && window.parent !== window) {
                    window.parent.postMessage({
                        type: 'HOLOCHATSTATS_CHAT_MESSAGE',
                        payload: data
                    }, parentOrigin);
                }
            }
        } catch (e) {
            console.error('Error processing message:', e);
        }
    }

    // Find and process messages in a container
    function scanForMessages(container) {
        if (!container) return;
        
        const selectors = [
            'yt-live-chat-text-message-renderer',
            'yt-live-chat-paid-message-renderer',
            'yt-live-chat-membership-item-renderer',
            'ytd-sponsorships-live-chat-gift-redemption-announcement-renderer',
            'yt-live-chat-paid-sticker-renderer'
        ];
        
        container.querySelectorAll(selectors.join(', ')).forEach(msg => {
            if (isValidChatMessage(msg)) {
                processMessage(msg);
            }
        });
    }

    // Start observing chat
    function startObserving() {
        if (observing) return;
        
        let chatContainer = document.querySelector('#items.style-scope.yt-live-chat-item-list-renderer');
        
        if (!chatContainer) {
            const selectors = [
                '#items.yt-live-chat-item-list-renderer',
                'yt-live-chat-item-list-renderer #items',
                '#item-offset #items'
            ];
            
            for (const selector of selectors) {
                chatContainer = document.querySelector(selector);
                if (chatContainer) break;
            }
        }
        
        if (!chatContainer) {
            setTimeout(startObserving, 1000);
            return;
        }

        observing = true;
        console.log('Observing chat messages');

        // Initial scan
        scanForMessages(chatContainer);

        // Set up mutation observer
        const observer = new MutationObserver(mutations => {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === 1) {
                        if (isValidChatMessage(node)) {
                            processMessage(node);
                        } else if (node.querySelector) {
                            scanForMessages(node);
                        }
                    }
                }
            }
        });

        observer.observe(chatContainer, { 
            childList: true, 
            subtree: true
        });
        
        notifyParent();
    }

    // Clean up periodically
    setInterval(() => {
        if (processedMessages.size > 5000) {
            processedMessages = new Set(Array.from(processedMessages).slice(-2500));
        }
    }, 60000);

    // Start observation
    setTimeout(startObserving, 1500);
    notifyParent();
})();