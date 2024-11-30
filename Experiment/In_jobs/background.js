const API_BASE_URL = 'https://api.linkedin.com/v2';
let accessToken = '';

// Set the alarm to run every 1 minute
chrome.alarms.create('checkJobs', { periodInMinutes: 1 });

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'checkJobs') {
    console.log('Alarm triggered, checking jobs');
    checkNewJobs();
  }
});

function checkNewJobs(manualJobRole = null, manualLocation = null) {
  console.log('Checking for new jobs');
  return new Promise((resolve, reject) => {
    chrome.storage.local.get(['jobRole', 'location'], function(result) {
      const jobRole = manualJobRole || result.jobRole || 'Data Scientist';
      const location = manualLocation || result.location || 'Bengaluru, Karnataka, India';

      console.log(`Searching for: ${jobRole} in ${location}`);
      const encodedJobRole = encodeURIComponent(jobRole);
      const encodedLocation = encodeURIComponent(location);
      const url = `https://www.linkedin.com/jobs/search/?keywords=${encodedJobRole}&location=${encodedLocation}`;

      chrome.tabs.create({ url: url, active: false }, (tab) => {
        chrome.tabs.onUpdated.addListener(function listener(tabId, info) {
          if (tabId === tab.id && info.status === 'complete') {
            chrome.tabs.onUpdated.removeListener(listener);
            console.log('LinkedIn page loaded, sending scrape message');
            setTimeout(() => {
              chrome.tabs.sendMessage(tab.id, { action: 'scrapeJobs' }, (response) => {
                console.log('Received response from content script:', response);
                if (chrome.runtime.lastError) {
                  console.error('Error:', chrome.runtime.lastError);
                  reject(chrome.runtime.lastError);
                } else if (response && response.jobs) {
                  resolve(response.jobs);
                }
                chrome.tabs.remove(tab.id);
              });
            }, 5000); // Increased delay to 5 seconds
          }
        });
      });
    });
  });
}

function processJobs(jobs, lastJobId) {
  return new Promise((resolve) => {
    chrome.storage.local.get(['jobs'], function(result) {
      const existingJobs = result.jobs || [];
      const newJobs = jobs.filter(job => !existingJobs.some(existingJob => existingJob.id === job.id));

      if (newJobs.length > 0) {
        newJobs.forEach(sendNotification);
        console.log(`Found ${newJobs.length} new job(s)`);
      } else {
        console.log('No new jobs found');
      }

      const updatedJobs = [...newJobs, ...existingJobs];
      chrome.storage.local.set({ jobs: updatedJobs, lastJobId: jobs[0]?.id });
      resolve({ jobs: updatedJobs, newJobs: newJobs });
    });
  });
}

function sendNotification(job) {
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icon.png',
    title: 'New Job Posting',
    message: `${job.title} at ${job.company}`,
  });
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'manualCheck') {
    console.log('Manual check requested');
    checkNewJobs(request.jobRole, request.location).then(
      (jobs) => {
        console.log('Jobs found:', jobs);
        chrome.storage.local.set({ jobs: jobs }, () => {
          sendResponse({ success: true, jobs: jobs });
        });
      },
      (error) => {
        console.error('Error during job check:', error);
        sendResponse({ success: false, error: error.message });
      }
    );
    return true; // Keeps the message channel open for asynchronous response
  }
});

// Load access token from storage on startup
chrome.storage.local.get(['accessToken'], (result) => {
  if (result.accessToken) {
    accessToken = result.accessToken;
  }
});
