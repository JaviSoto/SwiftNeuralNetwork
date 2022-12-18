//
//  NeuralNetworkView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct NeuralNetworkView: View {
    let trainingData: MNISTData

    @ObservedObject
    private var viewModel: NeuralNetworkViewModel

    init(trainingData: MNISTData) {
        self.trainingData = trainingData
        self.viewModel = NeuralNetworkViewModel(trainingData: trainingData)
    }

    @State
    private var randomItem: MNISTParser.DataSet.Item?

    @State
    private var randomItemPredictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome = .init()

    @State
    private var predictionOutcomeTableOrder = [KeyPathComparator(\ImageRecognitionNeuralNetwork.PredictionOutcome.Digit.confidence, order: .reverse)]

    var body: some View {
        NavigationView {
            NeuralNetworkConfigurationView(
                trainingData: trainingData,
                viewModel: viewModel
            )

            PredictionVisualizationView(
                item: randomItem,
                imageWidth: trainingData.training.imageWidth,
                updateImage: { updateImage() },
                predictionOutcome: $randomItemPredictionOutcome,
                tableOrder: $predictionOutcomeTableOrder
            )
        }
        .onAppear {
            updateImage()
        }
        .onChange(of: viewModel.state) { state in
            if state == .trained {
                updatePrediction()
            }
        }
        .onChange(of: randomItem?.image) { _ in
            updatePrediction()
        }
        .navigationSplitViewStyle(.prominentDetail)
        .frame(minWidth: 1000, minHeight: 600)
    }

    func updateImage() {
        randomItem = trainingData.all.items.randomElement()!
    }

    func updatePrediction() {
        if let randomItem {
            randomItemPredictionOutcome = viewModel.predictions(forImage: randomItem.image)
            randomItemPredictionOutcome.digits.sort(using: predictionOutcomeTableOrder)
        }
    }
}

#if DEBUG

struct NeuralNetworkView_Previews: PreviewProvider {
    private static let imageWidth: UInt32 = 28

    static var previews: some View {
        NeuralNetworkView(trainingData: .random)
    }
}

#endif
