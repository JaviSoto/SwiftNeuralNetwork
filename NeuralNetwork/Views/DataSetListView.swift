//
//  DataSetListView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct DataSetListView: View {
    let trainingData: MNISTData

    @State
    private var sortedItems: [MNISTParser.DataSet.Item] = []

    @Binding
    private(set) var selectedItemID: MNISTParser.DataSet.Item.Identifier?

    @Binding
    private(set) var sortOrder: [KeyPathComparator<MNISTParser.DataSet.Item>]

    private var imageWidth: UInt32 { trainingData.testing.imageWidth }

    var body: some View {
        Table(sortedItems.items, selection: $selectedItemID, sortOrder: $sortOrder) {
            TableColumn("ID", value: \.id) { item in
                Text(verbatim: item.id.description)
            }
            .width(80)

            TableColumn("Digit", value: \.label) { item in
                Text("\(item.label.representedNumber)")
            }
            .width(50)

            TableColumn("Image") { item in
                SampleImageView(sampleImage: item.image, width: imageWidth)
                    .frame(height: 50)
            }
        }
        .onChange(of: sortOrder) {
            updateSortedItems()
        }
        .onAppear() {
            updateSortedItems()
        }
    }

    private func updateSortedItems() {
        self.sortedItems = trainingData.all.sorted(using: sortOrder)
    }
}

struct DataSetListView_Previews: PreviewProvider {
    static var previews: some View {
        DataSetListView(trainingData: .random, selectedItemID: .constant(nil), sortOrder: .constant([]))
    }
}
