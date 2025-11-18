const searchBtn = document.getElementById('searchBtn');
const queryInput = document.getElementById('query');
const resultDiv = document.getElementById('result');

let syllabusData = {};

fetch('data.json')
  .then(response => response.json())
  .then(data => syllabusData = data)
  .catch(err => console.error('Error loading JSON:', err));

const keywordMap = {
  "3rd": "3rd sem ise", "third": "3rd sem ise",
  "4th": "4th sem ise", "fourth": "4th sem ise",
  "5th": "5th sem ise", "fifth": "5th sem ise",
  "6th": "6th sem ise", "sixth": "6th sem ise",
  "7th": "7th sem ise", "seventh": "7th sem ise",
  "svce": "svce bengaluru",
  "sri venkateshwara": "svce bengaluru"
};

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    searchBtn.click(); // Simulate button click
  }
});

// Trigger search on button click
searchBtn.addEventListener('click', async () => {
  const query = queryInput.value.trim().toLowerCase();
  if (!query) {
    resultDiv.innerHTML = "<p>Please enter a query.</p>";
    return;
  }


  let foundKey = null;
  for (let keyword in keywordMap) {
    if (query.includes(keyword)) {
      foundKey = keywordMap[keyword];
      break;
    }
  }

  if (foundKey && syllabusData[foundKey]) {
    const data = syllabusData[foundKey];
    let html = "";

    if (Array.isArray(data)) {
      html += `<h2>${foundKey.toUpperCase()}</h2>`;
      data.forEach(course => {
        html += `
          <div class="course">
            <h3>${course.code}: ${course.title}</h3>
            <p><strong>Credits:</strong> ${course.credits}</p>
            <details>
              <summary>View Units</summary>
              <ul>
                ${course.units.map(u => `<li><strong>Unit ${u.unit}:</strong> ${u.topics.join(", ")}</li>`).join("")}
              </ul>
            </details>
            <!-- View Important Questions -->
        ${course["Important questions"] ? `
        <details>
          <summary>View Important Questions</summary>
          <ul>
            ${course["Important questions"].map(q => `<li>${q}</li>`).join("")}
          </ul>
        </details>` : ''}
          </div>`;
      });
      
    } else {
      html += `
        <h2>${data.name}</h2>
        <p><strong>Location:</strong> ${data.location}</p>
        <p><strong>Established:</strong> ${data.established}</p>
        <p><strong>Affiliation:</strong> ${data.affiliation}</p>
        <p><strong>Programs:</strong> ${data.programs.join(", ")}</p>
        <p><strong>Website:</strong> <a href="${data.website}" target="_blank">${data.website}</a></p>
        <p>${data.info}</p>
      `;
    }

    resultDiv.innerHTML = html;
  } else {
    const fallback = await fetch(`https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json`)
      .then(res => res.json())
      .catch(() => null);

    if (fallback && fallback.AbstractText) {
      resultDiv.innerHTML = `<p>${fallback.AbstractText}</p>`;
    } else {
      resultDiv.innerHTML = `<p>No results found in JSON or DuckDuckGo.</p>`;
    }
  }
});
