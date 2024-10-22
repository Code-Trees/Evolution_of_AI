document.addEventListener('DOMContentLoaded', function() {
  const jobRoleInput = document.getElementById('jobRole');
  const locationInput = document.getElementById('location');
  const saveSettingsButton = document.getElementById('saveSettings');
  const checkNowButton = document.getElementById('checkNow');
  const statusElement = document.getElementById('status');
  const jobListElement = document.getElementById('jobList');

  chrome.storage.local.get(['jobRole', 'location', 'jobs'], function(result) {
    if (result.jobRole) jobRoleInput.value = result.jobRole;
    if (result.location) locationInput.value = result.location;
    if (result.jobs) displayJobs(result.jobs);
  });

  saveSettingsButton.addEventListener('click', function() {
    const jobRole = jobRoleInput.value.trim();
    const location = locationInput.value.trim();
    chrome.storage.local.set({ jobRole, location }, function() {
      statusElement.textContent = 'Settings saved!';
      setTimeout(() => { statusElement.textContent = ''; }, 3000);
    });
  });

  checkNowButton.addEventListener('click', function() {
    statusElement.textContent = 'Checking for jobs...';
    chrome.runtime.sendMessage({ 
      action: 'manualCheck',
      jobRole: jobRoleInput.value.trim(),
      location: locationInput.value.trim()
    }, function(response) {
      console.log('Received response from background:', response);
      if (chrome.runtime.lastError) {
        console.error('Runtime error:', chrome.runtime.lastError);
        statusElement.textContent = 'Error: Unable to check for jobs.';
      } else if (response && response.success) {
        statusElement.textContent = 'Job check completed!';
        displayJobs(response.jobs);
      } else {
        statusElement.textContent = 'Error checking for jobs.';
      }
    });
  });

  function displayJobs(jobs) {
    console.log('Displaying jobs:', jobs);
    jobListElement.innerHTML = '';
    if (jobs && jobs.length > 0) {
      jobs.forEach(job => {
        const jobElement = document.createElement('div');
        jobElement.className = 'job-item';
        jobElement.innerHTML = `
          <a href="${job.link}" target="_blank" class="job-title">${job.title}</a>
          <div class="job-company">${job.company}</div>
          <div class="job-location">${job.location}</div>
          <div class="job-metadata">${job.metadata}</div>
        `;
        jobListElement.appendChild(jobElement);
      });
    } else {
      jobListElement.innerHTML = '<p>No jobs found.</p>';
    }
  }
});
