//
//  NeuralNetworkView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct NeuralNetworkView: View {
    let trainingData: MNISTData

    @State
    private var dataListItems: [DataSetListView.Item] = []

    @State
    private var sortedDataListItems: [DataSetListView.Item] = []

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
    private var dataSetListTableOrder = [KeyPathComparator(\DataSetListView.Item.id, order: .forward)]

    @State
    private var columnVisibility: NavigationSplitViewVisibility = .all

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility, sidebar: {
            NeuralNetworkConfigurationView(
                trainingData: trainingData,
                viewModel: viewModel
            )
            .navigationSplitViewColumnWidth(min: 400, ideal: 450)
        }, content: {
            PredictionVisualizationView(
                item: selectedItem,
                imageWidth: trainingData.training.imageWidth,
                predictionOutcome: $randomItemPredictionOutcome,
                tableOrder: $predictionOutcomeTableOrder
            )
            .navigationSplitViewColumnWidth(min: 300, ideal: 400)
        }, detail: {
            DataSetListView(
                imageWidth: trainingData.all.imageWidth,
                sortedItems: sortedDataListItems,
                selectedItemID: $selectedItemID,
                sortOrder: $dataSetListTableOrder
            )
            .navigationSplitViewColumnWidth(min: 150, ideal: 200, max: 350)
        })
        .toolbar {
            ToolbarItem(placement: ToolbarItemPlacement.navigation) {
                SwiftUI.Label("Neural Network", systemImage: "brain")
            }
        }
        .onAppear {
            updatePredictions()
            selectedItemID = trainingData.all.items.randomElement()!.id
        }
        .onChange(of: viewModel.state) { state in
            if state == .trained {
                updatePredictions()
            }
        }
        .onChange(of: selectedItemID) { _ in
            updatePredictions()
        }
        .onChange(of: dataListItems) { _ in
            updateSortedDataListItems(using: dataSetListTableOrder)
        }
        .onChange(of: dataSetListTableOrder) { newOrder in
            updateSortedDataListItems(using: newOrder)
        }
        .frame(minWidth: 1000, minHeight: 600)
    }

    /// This is a workaround for SwiftUI being utter garbage and trying to use many GBs of RAM when we try to display ~10k elements.
    private static let maxItemsToDisplayInTable = 1000

    func updatePredictions() {
        if let selectedItem {
            randomItemPredictionOutcome = viewModel.predictions(forImage: selectedItem.image)
            randomItemPredictionOutcome.digits.sort(using: predictionOutcomeTableOrder)
        }

        dataListItems = measure("Calculate all predictions") {
            trainingData.testing.items.prefix(Self.maxItemsToDisplayInTable).map { item in
                return .init(
                    sample: item,
                    predictionOutcome: viewModel.predictions(forImage: item.image)
                )
            }
        }
    }

    private func updateSortedDataListItems(using order: [KeyPathComparator<DataSetListView.Item>]) {
        sortedDataListItems = dataListItems.sorted(using: order)
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
