pipeline {
  agent none
  stages {
    stage('test') {
      parallel {
        stage('test') {
          agent any
          steps {
            sh 'ls'
          }
        }

        stage('') {
          agent any
          steps {
            sh 'ls'
          }
        }

      }
    }

    stage('buld') {
      agent any
      steps {
        sh 'ls'
      }
    }

  }
}