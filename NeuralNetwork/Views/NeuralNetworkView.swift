//
//  NeuralNetworkView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct NeuralNetworkView: View {
    let trainingData: MNISTData

    private let itemsByID: [MNISTParser.DataSet.Item.Identifier: MNISTParser.DataSet.Item]

    @ObservedObject
    private var viewModel: NeuralNetworkViewModel

    init(trainingData: MNISTData) {
        self.trainingData = trainingData
        self.viewModel = NeuralNetworkViewModel(trainingData: trainingData)

        self.itemsByID = Dictionary(uniqueKeysWithValues: self.trainingData.all.items.map { ($0.id, $0) })
    }

    @State
    private var selectedItemID: MNISTParser.DataSet.Item.Identifier?

    private var selectedItem: MNISTParser.DataSet.Item? {
        return self.selectedItemID.map { itemsByID[$0]! }
    }

    @State
    private var randomItemPredictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome = .init()

    @State
    private var predictionOutcomeTableOrder = [KeyPathComparator(\ImageRecognitionNeuralNetwork.PredictionOutcome.Digit.confidence, order: .reverse)]

    @State
    private var dataSetListTableOrder = [KeyPathComparator(\MNISTParser.DataSet.Item.id, order: .reverse)]

    @State
    private var columnVisibility: NavigationSplitViewVisibility = .all

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility, sidebar: {
            NeuralNetworkConfigurationView(
                trainingData: trainingData,
                viewModel: viewModel
            )
        }, content: {
            PredictionVisualizationView(
                item: selectedItem,
                imageWidth: trainingData.training.imageWidth,
                predictionOutcome: $randomItemPredictionOutcome,
                tableOrder: $predictionOutcomeTableOrder
            )
        }, detail: {
            DataSetListView(
                trainingData: trainingData,
                selectedItemID: $selectedItemID,
                sortOrder: $dataSetListTableOrder
            )
            .navigationSplitViewColumnWidth(min: 100, ideal: 150, max: 200)
        })
        .toolbar {
            ToolbarItem(placement: ToolbarItemPlacement.navigation) {
                SwiftUI.Label("Neural Network", systemImage: "brain")
            }
        }
        .onAppear {
            selectedItemID = trainingData.all.items.randomElement()!.id
        }
        .onChange(of: viewModel.state) { state in
            if state == .trained {
                updatePrediction()
            }
        }
        .onChange(of: selectedItemID) { _ in
            updatePrediction()
        }
        .frame(minWidth: 1000, minHeight: 600)
    }

    func updatePrediction() {
        if let selectedItem {
            randomItemPredictionOutcome = viewModel.predictions(forImage: selectedItem.image)
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
