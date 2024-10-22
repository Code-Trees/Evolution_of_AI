function extractReferences() {
  // This is a very basic way to extract text that looks like references
  // It won't work for PDFs, but it will work for HTML pages with reference lists
  const bodyText = document.body.innerText;
  const lines = bodyText.split('\n');
  const references = lines.filter(line => 
    /^\[\d+\]/.test(line) || // Matches lines starting with [number]
    /^\d+\./.test(line)    // Matches lines starting with number.
  );
  return references;
}

function analyzeContent() {
  const references = extractReferences();
  if (references.length > 0) {
    chrome.runtime.sendMessage({action: 'foundReferences', references: references});
  }
}

// Run the analysis when the content script loads
analyzeContent();

// Send a message to clear references when the page is unloaded
window.addEventListener('unload', function() {
  chrome.runtime.sendMessage({action: 'clearReferences'});
});
