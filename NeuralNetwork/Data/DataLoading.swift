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
    static func loadTrainingData(maxCount: Int?) async -> MNISTData {
        return await Task.detached { () -> MNISTData in
            return measure("Loading training data") {
                let trainingImages = Bundle.main.url(forResource: "train-images-idx3-ubyte", withExtension: nil)!
                let trainingLabels = Bundle.main.url(forResource: "train-labels-idx1-ubyte", withExtension: nil)!

                let testImages = Bundle.main.url(forResource: "t10k-images-idx3-ubyte", withExtension: nil)!
                let testLabels = Bundle.main.url(forResource: "t10k-labels-idx1-ubyte", withExtension: nil)!

                final class Results: @unchecked Sendable {
                    var training: MNISTParser.DataSet!
                    var testing: MNISTParser.DataSet!
                }

                let results = Results()

                DispatchQueue.concurrentPerform(iterations: 2) { index in
                    switch index {
                    case 0:
                        results.training = try! MNISTParser.loadData(imageSetFileURL: trainingImages, labelDataFileURL: trainingLabels, category: .training, maxCount: maxCount)
                    case 1:
                        results.testing = try! MNISTParser.loadData(imageSetFileURL: testImages, labelDataFileURL: testLabels, category: .testing, maxCount: maxCount)
                    default: fatalError()
                    }
                }

                return .init(training: results.training, testing: results.testing, all: results.training + results.testing)
            }
        }.value
    }
}
