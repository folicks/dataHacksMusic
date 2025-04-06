document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('generate-form');
    const resultDiv = document.getElementById('result');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const seed = document.getElementById('seed').value;
        const length = document.getElementById('length').value;
        
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({seed, length})
            });
            
            const data = await response.json();
            resultDiv.textContent = data.lyrics;
        } catch (error) {
            console.error('Error:', error);
            resultDiv.textContent = 'Error generating lyrics';
        }
    });
});
