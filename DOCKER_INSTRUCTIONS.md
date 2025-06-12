# Docker Setup Instructions

## Building the Container

To build a container with the correct dependencies:

`ash
# Basic build
docker build -t witsv3:latest .

# Or with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t witsv3:latest .
`

## Running in Docker

To run WitsV3 in a Docker container:

`ash
# Interactive mode
docker run -it --rm witsv3:latest

# With volume mount for data persistence
docker run -it --rm -v /path/to/data:/app/data witsv3:latest

# As a background service
docker run -d --name witsv3-service witsv3:latest
`

## Using Docker Compose

`ash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
`

## Troubleshooting

If you encounter dependency issues in the container:

1. Try rebuilding with the --no-cache flag
2. Ensure the requirements.lock file is up to date
3. Check for platform-specific dependencies
