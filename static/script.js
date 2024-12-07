document.getElementById('questionForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const question = document.getElementById('question').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({ 'question': question })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('answer').textContent = `Answer: ${data.answer}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
