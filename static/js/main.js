// Function to handle form submissions and display predictions
function handlePrediction(formId, endpoint) {
    const form = document.getElementById(formId);
    const resultDiv = document.createElement('div');
    resultDiv.className = 'alert alert-info mt-3';
    form.appendChild(resultDiv);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                resultDiv.className = 'alert alert-danger mt-3';
                resultDiv.textContent = `Error: ${data.error}`;
            } else {
                resultDiv.className = 'alert alert-success mt-3';
                resultDiv.textContent = `Prediction: ${data.prediction}`;
                
                // Refresh the plot image
                const plotImg = document.querySelector('.plot-container img');
                if (plotImg) {
                    plotImg.src = plotImg.src + '?t=' + new Date().getTime();
                }
            }
        } catch (error) {
            resultDiv.className = 'alert alert-danger mt-3';
            resultDiv.textContent = `Error: ${error.message}`;
        }
    });
}

// Initialize form handlers when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // SLR form
    if (document.getElementById('slr-form')) {
        handlePrediction('slr-form', '/predict_slr');
    }
    
    // MLR form
    if (document.getElementById('mlr-form')) {
        handlePrediction('mlr-form', '/predict_mlr');
    }
    
    // LR form
    if (document.getElementById('lr-form')) {
        handlePrediction('lr-form', '/predict_lr');
    }
    
    // PR form
    if (document.getElementById('pr-form')) {
        handlePrediction('pr-form', '/predict_pr');
    }
    
    // KNN form
    if (document.getElementById('knn-form')) {
        handlePrediction('knn-form', '/predict_knn');
    }
}); 