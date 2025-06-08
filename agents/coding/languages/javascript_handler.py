"""
JavaScript language-specific code generation
"""

import json
import logging
from typing import List


class JavaScriptHandler:
    """Handles JavaScript-specific code generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def generate_main(self) -> str:
        """Generate JavaScript index.js"""
        return '''#!/usr/bin/env node

/**
 * Main application entry point
 */

const process = require('process');

// Main function
function main() {
    console.log('Application started');
    
    // Parse command line arguments
    const args = process.argv.slice(2);
    
    if (args.includes('--help') || args.includes('-h')) {
        showHelp();
        return;
    }
    
    if (args.includes('--version') || args.includes('-v')) {
        console.log('Version 1.0.0');
        return;
    }
    
    // Add your application logic here
    console.log('Hello, World!');
    console.log('Arguments:', args);
}

function showHelp() {
    console.log(`
Usage: node index.js [options]

Options:
  -h, --help     Show this help message
  -v, --version  Show version information
  
Examples:
  node index.js
  node index.js --help
    `);
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Run main function
if (require.main === module) {
    main();
}

module.exports = { main };
'''
    
    async def generate_web_app(self) -> str:
        """Generate JavaScript web app with Express"""
        return '''const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});

app.get('/api/data', (req, res) => {
    res.json({
        items: [
            { id: 1, name: 'Item 1', active: true },
            { id: 2, name: 'Item 2', active: false }
        ]
    });
});

// Error handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

module.exports = app;
'''
    
    async def generate_api(self) -> str:
        """Generate JavaScript API with Express"""
        return '''const express = require('express');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory database
let items = [];

// Routes
app.get('/', (req, res) => {
    res.json({ 
        message: 'API is running',
        version: '1.0.0',
        endpoints: ['/api/items', '/api/items/:id']
    });
});

app.get('/api/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date() });
});

// Get all items
app.get('/api/items', (req, res) => {
    res.json(items);
});

// Create item
app.post('/api/items', (req, res) => {
    const { name, description } = req.body;
    
    if (!name) {
        return res.status(400).json({ error: 'Name is required' });
    }
    
    const item = {
        id: uuidv4(),
        name,
        description: description || '',
        createdAt: new Date(),
        updatedAt: new Date()
    };
    
    items.push(item);
    res.status(201).json(item);
});

// Get single item
app.get('/api/items/:id', (req, res) => {
    const item = items.find(i => i.id === req.params.id);
    
    if (!item) {
        return res.status(404).json({ error: 'Item not found' });
    }
    
    res.json(item);
});

// Update item
app.put('/api/items/:id', (req, res) => {
    const itemIndex = items.findIndex(i => i.id === req.params.id);
    
    if (itemIndex === -1) {
        return res.status(404).json({ error: 'Item not found' });
    }
    
    const { name, description } = req.body;
    items[itemIndex] = {
        ...items[itemIndex],
        name: name || items[itemIndex].name,
        description: description !== undefined ? description : items[itemIndex].description,
        updatedAt: new Date()
    };
    
    res.json(items[itemIndex]);
});

// Delete item
app.delete('/api/items/:id', (req, res) => {
    const itemIndex = items.findIndex(i => i.id === req.params.id);
    
    if (itemIndex === -1) {
        return res.status(404).json({ error: 'Item not found' });
    }
    
    items.splice(itemIndex, 1);
    res.status(204).send();
});

// Error handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
    console.log(`API server running on http://localhost:${PORT}`);
});

module.exports = app;
'''
    
    async def generate_package_json(self, project_type: str) -> str:
        """Generate package.json for JavaScript projects"""
        base_package = {
            "name": "project",
            "version": "1.0.0",
            "description": "A JavaScript project",
            "main": "index.js",
            "scripts": {
                "start": "node index.js",
                "dev": "nodemon index.js",
                "test": "jest",
                "lint": "eslint .",
                "format": "prettier --write ."
            },
            "keywords": [],
            "author": "Your Name",
            "license": "MIT"
        }
        
        dependencies = {}
        dev_dependencies = {
            "nodemon": "^3.0.0",
            "jest": "^29.0.0",
            "eslint": "^8.0.0",
            "prettier": "^3.0.0"
        }
        
        if project_type in ['web_app', 'api']:
            dependencies.update({
                "express": "^4.18.0",
                "cors": "^2.8.5",
                "dotenv": "^16.0.0"
            })
        
        if project_type == 'web_app':
            dependencies.update({
                "ejs": "^3.1.0"
            })
        
        if project_type == 'api':
            dependencies.update({
                "uuid": "^9.0.0",
                "compression": "^1.7.0",
                "helmet": "^7.0.0"
            })
        
        base_package["dependencies"] = dependencies
        base_package["devDependencies"] = dev_dependencies
        
        return json.dumps(base_package, indent=2)
    
    async def generate_tests(self) -> str:
        """Generate JavaScript test file"""
        return '''const request = require('supertest');
const app = require('../index');

describe('Application Tests', () => {
    test('should export app', () => {
        expect(app).toBeDefined();
    });
    
    test('should have main function', () => {
        const { main } = require('../index');
        expect(typeof main).toBe('function');
    });
});

describe('API Tests', () => {
    test('GET / should return welcome message', async () => {
        const response = await request(app)
            .get('/')
            .expect(200);
        
        expect(response.body).toHaveProperty('message');
    });
    
    test('GET /api/health should return health status', async () => {
        const response = await request(app)
            .get('/api/health')
            .expect(200);
        
        expect(response.body).toHaveProperty('status', 'healthy');
        expect(response.body).toHaveProperty('timestamp');
    });
    
    test('GET /api/items should return array', async () => {
        const response = await request(app)
            .get('/api/items')
            .expect(200);
        
        expect(Array.isArray(response.body)).toBe(true);
    });
    
    test('POST /api/items should create new item', async () => {
        const newItem = { name: 'Test Item', description: 'Test Description' };
        
        const response = await request(app)
            .post('/api/items')
            .send(newItem)
            .expect(201);
        
        expect(response.body).toHaveProperty('id');
        expect(response.body.name).toBe(newItem.name);
    });
});

// Mock data for testing
const mockData = {
    items: [
        { id: '1', name: 'Item 1' },
        { id: '2', name: 'Item 2' }
    ]
};

describe('Unit Tests', () => {
    test('should process mock data', () => {
        expect(mockData.items).toHaveLength(2);
        expect(mockData.items[0].name).toBe('Item 1');
    });
});
'''