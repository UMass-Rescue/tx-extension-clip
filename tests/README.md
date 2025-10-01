# Running Tests

## Prerequisites

- [Docker](httpss://www.docker.com/get-started) must be installed and running.

## Instructions

1.  **Build the Docker Image**

    From the project root, build the test image:
    ```sh
    docker build -t clip-test-runner .
    ```

2.  **Run the Tests**

    Execute the tests using the built image:
    ```sh
    docker run --rm -e TX_CLIP_TEST_ENV=true clip-test-runner
    ```
