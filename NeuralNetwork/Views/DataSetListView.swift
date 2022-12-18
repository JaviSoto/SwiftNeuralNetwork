//
//  DataSetListView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct DataSetListView: View {
    struct Item: Equatable, Identifiable {
        let sample: MNISTParser.DataSet.Item
        let predictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome

        var predictionCorrectness: PredictionCorrectness {
            .init(predictionOutcome: predictionOutcome, expectedLabel: sample.label)
        }

        var id: MNISTParser.DataSet.Item.Identifier { sample.id }
    }

    let imageWidth: UInt32
    let sortedItems: [Item]

    @Binding
    private(set) var selectedItemID: Item.ID?

    @Binding
    private(set) var sortOrder: [KeyPathComparator<Item>]

    var body: some View {
        Table(sortedItems, selection: $selectedItemID, sortOrder: $sortOrder) {
            TableColumn("ID", value: \.sample.id) { item in
                Text("\(item.sample.id.value)")
            }
            .width(ideal: 25)

            TableColumn("Prediction", value: \.predictionCorrectness) { item in
                PredictionCorrectnessView(correctness: item.predictionCorrectness)
            }
            .width(60)

            TableColumn("Digit", value: \.sample.label) { item in
                Text("\(item.sample.label.representedNumber)")
            }
            .width(50)

            TableColumn("Image") { item in
                SampleImageView(sampleImage: item.sample.image, width: imageWidth)
                    .frame(height: 50)
            }
            .width(50)
        }
    }
}

#if DEBUG

struct DataSetListView_Previews: PreviewProvider {
    static var allItems: [DataSetListView.Item] {
        return MNISTData.random.all.items.map { item in
                .init(sample: item, predictionOutcome: .init())
        }
    }
    static var previews: some View {
        DataSetListView(imageWidth: 28, sortedItems: allItems, selectedItemID: .constant(nil), sortOrder: .constant([]))
    }
}

#endif
