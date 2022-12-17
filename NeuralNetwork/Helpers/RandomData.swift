//
//  RandomData.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import Foundation

#if DEBUG

extension MNISTParser.DataSet {
    static let imageWidth: UInt32 = 28

    static var randomItem: MNISTParser.DataSet.Item {
        return (
            image: .init(pixels: Array(0..<(imageWidth * imageWidth)).map { _ in UInt8.random(in: 0...UInt8.max) }),
            label: .init(representedNumber: (UInt8(0)...9).randomElement()!)
        )
    }
}

extension MNISTData {
    static var random: MNISTData {
        return .init(
            training: .init(imageWidth: MNISTParser.DataSet.imageWidth, items: (0..<10).map { _ in MNISTParser.DataSet.randomItem }),
            testing: .init(imageWidth: MNISTParser.DataSet.imageWidth, items: (0..<10).map { _ in MNISTParser.DataSet.randomItem }),
            all: .init(imageWidth: MNISTParser.DataSet.imageWidth, items: (0..<20).map { _ in MNISTParser.DataSet.randomItem })
        )
    }
}

#endif
