document.addEventListener('DOMContentLoaded', function() {
  try {
    const keywordInput = document.getElementById('keywordInput');
    const addKeywordButton = document.getElementById('addKeyword');
    const keywordList = document.getElementById('keywordList');
    const searchButton = document.getElementById('searchButton');
    const results = document.getElementById('results');
    const referenceList = document.getElementById('referenceList');

    // Load saved keywords
    chrome.storage.sync.get(['keywords'], function(data) {
      const keywords = data.keywords || [];
      updateKeywordList(keywords);
    });

    // Add keyword
    addKeywordButton.addEventListener('click', function() {
      const keyword = keywordInput.value.trim();
      if (keyword) {
        chrome.storage.sync.get(['keywords'], function(data) {
          const keywords = data.keywords || [];
          keywords.push(keyword);
          chrome.storage.sync.set({keywords: keywords}, function() {
            updateKeywordList(keywords);
            keywordInput.value = '';
          });
        });
      }
    });

    // Search button
    searchButton.addEventListener('click', function() {
      console.log('Search button clicked');
      results.innerHTML = 'Searching...';
      chrome.runtime.sendMessage({action: 'searchArxiv'}, function(response) {
        console.log('Received response:', response);
        if (response && response.html) {
          const parser = new DOMParser();
          const doc = parser.parseFromString(response.html, 'text/html');
          const papers = doc.querySelectorAll('li.arxiv-result');
          let resultsHtml = '<h3>Search Results:</h3>';

          papers.forEach(paper => {
            const title = paper.querySelector('p.title').textContent.trim();
            const authors = paper.querySelector('p.authors').textContent.trim();
            const abstract = paper.querySelector('span.abstract-full').textContent.trim();
            const link = paper.querySelector('p.list-title a').href;

            resultsHtml += `<div>
              <h4><a href="${link}" target="_blank">${title}</a></h4>
              <p>${authors}</p>
              <p>${abstract}</p>
            </div><hr>`;
          });

          results.innerHTML = resultsHtml || 'No results found.';
        } else {
          results.innerHTML = (response && response.results) || 'An error occurred while searching Arxiv.';
        }
      });
    });

    // Function to display references
    function displayReferences(references) {
      const referenceList = document.getElementById('referenceList');
      if (references && references.length > 0) {
        referenceList.innerHTML = '<h3>References in current page:</h3>';
        references.forEach(ref => {
          const li = document.createElement('li');
          li.textContent = ref;
          referenceList.appendChild(li);
        });
      } else {
        referenceList.innerHTML = ''; // Clear the reference list if there are no references
      }
    }

    // Check for stored references each time the popup is opened
    chrome.storage.local.get(['currentReferences'], function(data) {
      displayReferences(data.currentReferences);
    });

    function updateKeywordList(keywords) {
      keywordList.innerHTML = '<h3>Saved Keywords:</h3>';
      keywords.forEach(function(keyword, index) {
        const keywordElement = document.createElement('div');
        keywordElement.textContent = keyword;
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.addEventListener('click', function() {
          keywords.splice(index, 1);
          chrome.storage.sync.set({keywords: keywords}, function() {
            updateKeywordList(keywords);
          });
        });
        keywordElement.appendChild(removeButton);
        keywordList.appendChild(keywordElement);
      });
    }
  } catch (error) {
    console.error('An error occurred in popup.js:', error);
    document.body.innerHTML = '<p>An error occurred. Please check the console and reload.</p>';
  }
});
