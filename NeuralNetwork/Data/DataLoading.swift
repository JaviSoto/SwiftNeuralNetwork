//
//  DataLoading.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import Foundation

struct MNISTData {
    let training: MNISTParser.DataSet
    let testing: MNISTParser.DataSet

    let all: MNISTParser.DataSet
}

enum DataLoading {
    static func loadTrainingData() async -> MNISTData {
        return await Task.detached { () -> MNISTData in
            return measure("Loading training data") {
                let trainingImages = Bundle.main.url(forResource: "train-images-idx3-ubyte", withExtension: nil)!
                let trainingLabels = Bundle.main.url(forResource: "train-labels-idx1-ubyte", withExtension: nil)!

                let testImages = Bundle.main.url(forResource: "t10k-images-idx3-ubyte", withExtension: nil)!
                let testLabels = Bundle.main.url(forResource: "t10k-labels-idx1-ubyte", withExtension: nil)!

    #if DEBUG
                let maxCount = 100
    #else
                let maxCount: Int? = nil
    #endif

                let training = try! MNISTParser.loadData(imageSetFileURL: trainingImages, labelDataFileURL: trainingLabels, category: .training, maxCount: maxCount)
                let testing = try! MNISTParser.loadData(imageSetFileURL: testImages, labelDataFileURL: testLabels, category: .testing, maxCount: maxCount)

                return .init(training: training, testing: testing, all: training + testing)
            }
        }.value
    }
}
