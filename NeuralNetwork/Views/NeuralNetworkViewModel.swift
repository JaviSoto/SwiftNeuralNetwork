//
//  NeuralNetworkViewModel.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import Foundation

@MainActor
final class NeuralNetworkViewModel: ObservableObject {
    private let trainingData: MNISTData

    @Published
    var neuralNetwork: ImageRecognitionNeuralNetwork

    enum State: Equatable {
        case idle
        case training
        case trained
    }

    @Published
    private(set) var state: State = .idle

    @Published
    private(set) var trainingDataAccuracy: Double = 0

    @Published
    private(set) var testDataAccuracy: Double = 0

    @Published
    private(set) var trainingSessionsAccuracies: [NeuralNetwork.TrainingSessionAccuracyProgress] = []

    private let trainingProgressObserver = NeuralNetwork.TrainingProgressObserver()

    init(trainingData: MNISTData) {
        self.trainingData = trainingData
        self.neuralNetwork = ImageRecognitionNeuralNetwork(trainingData: trainingData.training)
        self.updateAccuracies()
    }

    func train() {
        self.state = .training

        let trainingSessionIndex = trainingSessionsAccuracies.count
        trainingSessionsAccuracies.append([])

        let token = trainingProgressObserver.$accuracies
            .receive(on: DispatchQueue.main)
            .sink { [weak self] accuracies in
                self?.trainingSessionsAccuracies[trainingSessionIndex] = accuracies
            }

        Task {
            var neuralNetwork = neuralNetwork

            await neuralNetwork.trainAsync(with: trainingProgressObserver)

            await MainActor.run { [neuralNetwork] in
                self.neuralNetwork = neuralNetwork
                self.updateAccuracies()
                self.state = .trained

                token.cancel()
            }
        }
    }

    func predictions(forImage image: SampleImage) -> ImageRecognitionNeuralNetwork.PredictionOutcome {
        return neuralNetwork.digitPredictions(withInputImage: image)
    }

    private func updateAccuracies() {
        measure("Calculating accuracy") {
            let (trainingImages, trainingLabels) = trainingData.training.trainingAndValidationMatrixes
            trainingDataAccuracy = neuralNetwork.neuralNetwork.accuracy(usingInputData: trainingImages, expectedOutput: trainingLabels)

            let (testImages, testLabels) = trainingData.testing.trainingAndValidationMatrixes
            testDataAccuracy = neuralNetwork.neuralNetwork.accuracy(usingInputData: testImages, expectedOutput: testLabels)
        }
    }
}
