// Book Writing GUI JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize event listeners
    initEventListeners();
    
    // Load books on page load
    loadBooks();
});

function showError(message) {
    const errorElement = document.getElementById('error-message');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        setTimeout(() => {
            errorElement.style.display = 'none';
        }, 5000);
    } else {
        alert('Error: ' + message);
    }
}

function showSuccess(message) {
    const successElement = document.getElementById('success-message');
    if (successElement) {
        successElement.textContent = message;
        successElement.style.display = 'block';
        setTimeout(() => {
            successElement.style.display = 'none';
        }, 5000);
    } else {
        alert('Success: ' + message);
    }
}

function loadBooks() {
    // Fetch books for dropdowns and book list
    console.log('Loading books...');
    fetch('/api/books/')
        .then(response => {
            console.log('Books API response status:', response.status);
            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }
            return response.json();
        })
        .then(books => {
            console.log('Books loaded:', books);
            populateBookDropdowns(books);
            populateBooksContainer(books);
        })
        .catch(error => {
            console.error('Error loading books:', error);
            showError('Failed to load books. Please refresh the page. Error: ' + error.message);
        });
}

function populateBooksContainer(books) {
    const booksContainer = document.getElementById('books-container');
    if (!booksContainer) {
        console.error('Books container element not found');
        return;
    }
    
    // Clear existing content
    booksContainer.innerHTML = '';
    
    if (!books || books.length === 0) {
        console.log('No books found');
        booksContainer.innerHTML = '<p class="empty-message">No books found. Create your first book above!</p>';
        return;
    }
    
    console.log(`Displaying ${books.length} books`);
    
    // Add each book to the container
    books.forEach(book => {
        const bookCard = document.createElement('div');
        bookCard.className = 'book-card';
        bookCard.innerHTML = `
            <h3 class="book-title">${book.title}</h3>
            <p><strong>Author:</strong> ${book.author}</p>
            <p><strong>Genre:</strong> ${book.genre || 'Not specified'}</p>
            <p>${book.description || 'No description available.'}</p>
            <div class="book-stats">
                <span>${book.chapters ? book.chapters.length : 0} chapters</span>
                <span>${book.characters ? book.characters.length : 0} characters</span>
            </div>
        `;
        booksContainer.appendChild(bookCard);
    });
}

function populateBookDropdowns(books) {
    const dropdowns = [
        document.getElementById('chapter-book'),
        document.getElementById('chapters-book-filter'),
        document.getElementById('character-book'),
        document.getElementById('characters-book-filter'),
        document.getElementById('generation-book'),
        document.getElementById('research-book')
    ];
    
    dropdowns.forEach(dropdown => {
        if (dropdown) {
            // Clear existing options
            dropdown.innerHTML = '';
            
            // Add placeholder if it's not a filter dropdown
            if (!dropdown.id.includes('filter')) {
                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = 'Select a book';
                placeholder.disabled = true;
                placeholder.selected = true;
                dropdown.appendChild(placeholder);
            }
            
            // Add book options
            books.forEach(book => {
                const option = document.createElement('option');
                option.value = book.id;
                option.textContent = book.title;
                dropdown.appendChild(option);
            });
        }
    });
}

function initEventListeners() {
    // Book form submission
    const bookForm = document.getElementById('book-form');
    if (bookForm) {
        bookForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitBookForm();
        });
    }
}

function submitBookForm() {
    const form = document.getElementById('book-form');
    const formData = new FormData(form);
    
    // Log the form data to help debug
    console.log('Form data entries:');
    for (let pair of formData.entries()) {
        console.log(pair[0] + ': ' + pair[1]);
    }
    
    // Create the book data object with the correct field names
    const bookData = {
        title: formData.get('title'),
        author: formData.get('author'),
        genre: formData.get('genre') || 'Fiction', // Default to Fiction if empty
        description: formData.get('description') || '' // Default to empty string if not provided
    };
    
    // Validate required fields
    if (!bookData.title || !bookData.author) {
        showError('Title and Author are required fields.');
        return;
    }
    
    console.log('Sending book data:', bookData);
    
    fetch('/api/books/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(bookData)
    })
    .then(response => {
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (!response.ok) {
            // If the response is not OK, throw an error with the status
            return response.text().then(text => {
                console.error('Error response body:', text);
                throw new Error(`Server responded with status ${response.status}: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Book created:', data);
        showSuccess('Book created successfully!');
        form.reset();
        
        // Refresh the book list
        loadBooks();
    })
    .catch(error => {
        console.error('Error creating book:', error);
        showError('Failed to create book. Please try again. Error: ' + error.message);
    });
}
