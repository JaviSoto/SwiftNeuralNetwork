//
//  ImageRenderingTests.swift
//  NeuralNetworkTests
//
//  Created by Javier Soto on 12/18/22.
//

import XCTest
@testable import NeuralNetwork

final class ImageRenderingTests: XCTestCase {
    func testParsingCGImage() async {
        let data = await DataLoading.loadTrainingData(maxCount: 1)
        let image = data.all.items[0].image
        let cgImage = image.pixels.map(\.colorPixel).asCGImage(width: image.width)

        let sampleImageBack = SampleImage(cgImage)

        XCTAssertEqual(sampleImageBack.pixels, image.pixels)
    }
}
