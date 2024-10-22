chrome.runtime.onInstalled.addListener(function() {
  chrome.storage.sync.set({keywords: []});
  console.log('Extension installed');
});

chrome.tabs.onCreated.addListener(function(tab) {
  console.log('New tab created, searching arXiv');
  searchArxiv();
});

// Listen for tab close events
chrome.tabs.onRemoved.addListener(function(tabId, removeInfo) {
  // Clear the stored references when a tab is closed
  chrome.storage.local.remove('currentReferences', function() {
    console.log('References cleared for closed tab');
  });
});

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  console.log('Message received:', request);
  if (request.action === 'searchArxiv') {
    searchArxiv(sendResponse);
    return true; // Indicates that the response will be sent asynchronously
  } else if (request.action === 'searchReferences') {
    searchReferences(request.references, sendResponse);
    return true;
  } else if (request.action === 'foundReferences') {
    // Store the found references in chrome.storage
    chrome.storage.local.set({currentReferences: request.references});
  } else if (request.action === 'clearReferences') {
    // Clear the stored references
    chrome.storage.local.remove('currentReferences', function() {
      console.log('References cleared');
    });
  }
});

function searchArxiv(sendResponse) {
  console.log('Searching arXiv');
  chrome.storage.sync.get(['keywords'], function(data) {
    const keywords = data.keywords || [];
    console.log('Keywords:', keywords);
    if (keywords.length === 0) {
      console.log('No keywords found');
      if (sendResponse) sendResponse({results: 'No keywords saved. Please add keywords to search.'});
      return;
    }

    const searchQuery = keywords.join('+');
    const arxivUrl = `https://arxiv.org/search/cs?query=${searchQuery}&searchtype=all&abstracts=show&order=-announced_date_first&size=50`;
    console.log('Fetching from URL:', arxivUrl);

    fetch(arxivUrl)
      .then(response => response.text())
      .then(html => {
        console.log('Received HTML response');
        if (sendResponse) sendResponse({html: html});
      })
      .catch(error => {
        console.error('Error:', error);
        if (sendResponse) sendResponse({results: 'An error occurred while searching Arxiv.'});
      });
  });
}

function searchReferences(references, sendResponse) {
  console.log('Searching references:', references);
  // Implement the search logic here, similar to searchArxiv
  // You might want to search for each reference individually
  // and compile the results
}
