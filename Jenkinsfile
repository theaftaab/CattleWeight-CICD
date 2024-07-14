pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub')
        DOCKERHUB_REPO = 'theaftaab/CattleWeight'
    }

    stages {
        stage('Checkout') {
            steps {
                // Checkout the source code from the repository
                git branch: 'main', url: 'https://github.com/theaftaab/CattleWeight.git'
            }
        }

        stage('Build') {
            steps {
                // Build the Python project (e.g., install dependencies, run tests)
                sh '''
                export PYTHON3=/Users/aftaabhussain/anaconda3/bin/python3
                python3 -m venv venv
                source venv/bin/activate
                pip install -r requirements.txt
                pytest
                '''
            }
        }

        stage('Docker Build') {
            steps {
                // Build the Docker image
                sh 'docker build -t $DOCKERHUB_REPO:latest .'
            }
        }

//         stage('Docker Login') {
//             steps {
//                 // Log in to Docker Hub
//                 sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
//             }
//         }

        stage('Docker Push') {
            steps {
                // Push the Docker image to Docker Hub
                sh 'docker push $DOCKERHUB_REPO:latest'
            }
        }
    }

    post {
        always {
            // Clean up the Docker images to free space
            sh 'docker rmi $DOCKERHUB_REPO:latest'
        }
    }
}