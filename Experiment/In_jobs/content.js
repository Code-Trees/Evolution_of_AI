chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'scrapeJobs') {
    console.log('Received scrape request');
    const jobs = scrapeJobListings();
    console.log('Scraped jobs:', jobs);
    sendResponse({ jobs: jobs });
  }
  return true; // Keeps the message channel open for asynchronous response
});

function scrapeJobListings() {
  console.log('Starting job scraping');
  const jobCards = document.querySelectorAll('.jobs-search__results-list > li');
  console.log('Found job cards:', jobCards.length);
  
  return Array.from(jobCards).map(card => {
    const titleElement = card.querySelector('h3.base-search-card__title');
    const companyElement = card.querySelector('h4.base-search-card__subtitle');
    const locationElement = card.querySelector('.job-search-card__location');
    const linkElement = card.querySelector('a.base-card__full-link');
    const metadataElements = card.querySelectorAll('.base-search-card__metadata span');

    const title = titleElement ? titleElement.textContent.trim() : 'N/A';
    const company = companyElement ? companyElement.textContent.trim() : 'N/A';
    const location = locationElement ? locationElement.textContent.trim() : 'N/A';
    const link = linkElement ? linkElement.href : '#';
    const metadata = Array.from(metadataElements).map(el => el.textContent.trim()).join(' • ');

    console.log('Scraped job:', { title, company, location, link, metadata });
    return { title, company, location, link, metadata };
  });
}

function getAdditionalInfo(card) {
  const infoElements = card.querySelectorAll('.job-search-card__job-insight');
  return Array.from(infoElements).map(el => el.textContent.trim()).join(' • ');
}
