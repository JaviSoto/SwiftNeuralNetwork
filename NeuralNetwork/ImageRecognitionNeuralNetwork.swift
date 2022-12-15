//
//  ImageRecognitionNeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/15/22.
//

import Foundation
import SwiftMatrix

struct ImageRecognitionNeuralNetwork {
    private var neuralNetwork: NeuralNetwork

    private let trainingData: MNISTParser.DataSet

    init(trainingData: MNISTParser.DataSet) {
        self.trainingData = trainingData

        self.neuralNetwork = NeuralNetwork(
            inputLayerNeuronCount: Int(trainingData.imageWidth * trainingData.imageWidth),
            outputLayerSize: 10
        )
        neuralNetwork.addHiddenLayer(withNeuronCount: 10, activationFunction: .reLU)
        neuralNetwork.addHiddenLayer(withNeuronCount: 10, activationFunction: .softMax)
    }

    mutating func train() {
        let (training, validation) = trainingData.trainingAndValidationMatrixes

        neuralNetwork.train(
            usingTrainingData: training,
            validationData: validation,
            iterations: 500,
            alpha: 0.1
        )
    }

    func digitPredictions(withInputImage image: SampleImage) -> Matrix {
        return neuralNetwork.predictions(usingData: image.normalizedPixelVector)
    }
}

private extension SampleImage {
    var normalizedPixelVector: [Double] {
        return pixels.map { Double($0) / Double(SampleImage.Pixel.max) }
    }
}

extension MNISTParser.DataSet {
    var trainingAndValidationMatrixes: (training: Matrix, validation: Matrix) {
        let training = Matrix(self.items.map { $0.image.normalizedPixelVector })′
        let validation = Matrix(self.items.map { [Double($0.label.representedNumber)] })

        return (training, validation)
    }
}

private extension Matrix {
    init(_ sampleSet: MNISTParser.DataSet) {
        self = Matrix(sampleSet.items.map { item in
            item.image.normalizedPixelVector
        })
    }
}
