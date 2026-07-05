const nav = document.getElementById('navbar');
const themeToggle = document.getElementById('themeToggle');
const root = document.documentElement;

function updateTheme(theme) {
    root.setAttribute('data-theme', theme);
    nav.classList.toggle('navbar-dark', theme === 'dark');
    nav.classList.toggle('navbar-light', theme === 'light');
    localStorage.setItem('theme', theme);
}

function getSavedTheme() {
    return localStorage.getItem('theme') || 'dark';
}

function toggleTheme() {
    const currentTheme = root.getAttribute('data-theme');
    updateTheme(currentTheme === 'light' ? 'dark' : 'light');
}

if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
}

updateTheme(getSavedTheme());

// Dropzone configuration
Dropzone.autoDiscover = false;

const dropzoneElement = document.getElementById('dropzone');
const dropzone = new Dropzone(dropzoneElement, {
    url: '/predict',
    autoProcessQueue: false,
    uploadMultiple: false,
    maxFiles: 1,
    maxFilesize: 10, // MB
    acceptedFiles: 'image/jpeg, image/png, image/bmp, image/webp, .jpg, .jpeg, .png, .bmp, .webp',
    clickable: true,
    dictDefaultMessage: '',
    paramName: 'image'
});

// Handle drag over
dropzone.on('dragenter', () => {
    dropzoneElement.classList.add('dragging');
});

dropzone.on('dragleave', () => {
    dropzoneElement.classList.remove('dragging');
});

dropzone.on('dragend', () => {
    dropzoneElement.classList.remove('dragging');
});

// Handle file additions
dropzone.on('addedfile', (file) => {
    dropzoneElement.classList.remove('dragging');
    // Automatically submit the file
    submitImage(file);
});

// Handle file errors (validation)
dropzone.on('error', (file, errorMessage) => {
    let displayMessage = errorMessage;
    
    if (errorMessage.includes('larger than')) {
        displayMessage = 'File size exceeds 10MB limit';
    } else if (errorMessage.includes('not accepted')) {
        displayMessage = 'File type not supported. Please upload an image file.';
    }
    
    showError(displayMessage);
});

// Submit image file
function submitImage(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    // Show loading state
    dropzoneElement.style.opacity = '0.5';
    dropzoneElement.style.pointerEvents = 'none';
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        // Try to parse response as JSON
        return response.json().then(data => {
            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze image');
            }
            return data;
        }).catch(error => {
            // If JSON parsing fails, check if it's a JSON error or server error
            if (!response.ok) {
                throw new Error('Server error: ' + response.status + ' ' + response.statusText);
            }
            // If response was ok but JSON failed, something else went wrong
            throw new Error('Failed to parse server response');
        });
    })
    .then(data => {
        displayResults(data);
        dropzone.removeFile(file);
    })
    .catch(error => {
        showError(error.message);
        dropzone.removeFile(file);
    })
    .finally(() => {
        dropzoneElement.style.opacity = '1';
        dropzoneElement.style.pointerEvents = 'auto';
    });
}

// Display results
function displayResults(data) {
    const scoreRing = document.getElementById('scoreRing');
    const resultTitle = document.getElementById('resultTitle');
    const resultSummary = document.getElementById('resultSummary');
    const resultStack = document.querySelector('.result-stack');
    
    // Determine title and ring probability based on the top class
    let topClass = null;
    let topProbability = 0;
    
    if (data.all_classes && Array.isArray(data.all_classes) && data.all_classes.length > 0) {
        topClass = data.all_classes[0].class;
        topProbability = data.all_classes[0].probability;
    }
    
    // Update score ring with top probability
    scoreRing.style.setProperty('--percentage', topProbability);
    
    // Update score value text
    const scoreValue = scoreRing.querySelector('span') || document.createElement('span');
    scoreValue.textContent = topProbability + '%';
    if (!scoreRing.querySelector('span')) {
        scoreRing.appendChild(scoreValue);
    }
    
    // Set title based on top class
    if (topClass === "Real") {
        if (topProbability > 70) {
            resultTitle.textContent = 'Likely a real image';
        } else if (topProbability > 50) {
            resultTitle.textContent = 'Possibly real';
        } else {
            resultTitle.textContent = 'Possibly AI-generated';
        }
    } else {
        if (topProbability > 70) {
            resultTitle.textContent = 'Likely AI-generated';
        } else if (topProbability > 50) {
            resultTitle.textContent = 'Possibly AI-generated';
        } else {
            resultTitle.textContent = 'Possibly real';
        }
    }
    
    // Update summary
    resultSummary.textContent = `The detector classifies this image as "${data.predicted_class}" with ${data.class_confidence}% confidence.`;
    
    // Clear and populate result buttons with top 3 classes
    resultStack.innerHTML = '';
    
    if (data.all_classes && Array.isArray(data.all_classes)) {
        data.all_classes.forEach(cls => {
            const button = document.createElement('button');
            button.className = 'button-bubble' + (cls.class === data.predicted_class ? ' active' : '');
            button.type = 'button';
            button.textContent = `${cls.class} - ${cls.probability}%`;
            resultStack.appendChild(button);
        });
    }
}

// Show error message
function showError(message) {
    // Create a simple alert or toast notification
    const dropzoneMessage = dropzoneElement.querySelector('.dz-message');
    if (dropzoneMessage) {
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            padding: 1rem;
            background: rgba(255, 100, 100, 0.2);
            border: 1px solid rgba(255, 100, 100, 0.5);
            border-radius: 8px;
            color: #ff6464;
            margin-bottom: 1rem;
            animation: fadeIn 300ms ease;
        `;
        errorDiv.textContent = 'Error: ' + message;
        dropzoneMessage.insertBefore(errorDiv, dropzoneMessage.firstChild);
        
        // Remove error after 5 seconds
        setTimeout(() => {
            errorDiv.style.animation = 'fadeOut 300ms ease';
            setTimeout(() => errorDiv.remove(), 300);
        }, 5000);
    }
}

// Demo image buttons
document.querySelectorAll('.floating-demo').forEach(button => {
    button.addEventListener('click', async () => {
        // Update active state
        document.querySelectorAll('.floating-demo').forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        
        // Get image path from img src
        const imgElement = button.querySelector('img');
        if (!imgElement) return;
        
        const imagePath = imgElement.src;
        
        // Fetch the image and convert to File
        try {
            const response = await fetch(imagePath);
            const blob = await response.blob();
            
            // Create a File object from the blob
            const filename = imagePath.split('/').pop() || 'demo-image.png';
            const file = new File([blob], filename, { type: blob.type });
            
            // Submit the image using the existing submitImage function
            submitImage(file);
        } catch (error) {
            showError('Failed to load demo image: ' + error.message);
        }
    });
});

// Layer button event listeners
document.querySelectorAll('.layer-grid .button-bubble').forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all buttons
        document.querySelectorAll('.layer-grid .button-bubble').forEach(b => b.classList.remove('active'));
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Update layer card information
        const layerTitle = document.getElementById('layerTitle');
        const layerSummary = document.getElementById('layerSummary');
        const layerAccent = document.getElementById('layerAccent');
        
        layerTitle.textContent = button.getAttribute('data-layer');
        layerSummary.textContent = button.getAttribute('data-summary');
        layerAccent.textContent = button.getAttribute('data-accent');
    });
});

// Load demo image on page load
window.addEventListener('load', () => {
    const demo = document.querySelector('.floating-demo.active');

    if (demo) {
        // Trigger click on the first demo button to load and display results
        demo.click();
    }
});