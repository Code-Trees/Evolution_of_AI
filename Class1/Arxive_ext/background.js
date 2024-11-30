chrome.runtime.onInstalled.addListener(function() {
  chrome.storage.sync.set({keywords: []});
  console.log('Extension installed');
});

chrome.tabs.onCreated.addListener(function(tab) {
  console.log('New tab created, searching arXiv');
  searchArxiv();
});

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  console.log('Message received:', request);
  if (request.action === 'searchArxiv') {
    searchArxiv(sendResponse);
    return true; // Indicates that the response will be sent asynchronously
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
