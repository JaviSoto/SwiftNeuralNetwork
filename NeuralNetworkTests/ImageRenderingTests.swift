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

    func testCroppingAndReCentering() async {
        let data = await DataLoading.loadTrainingData(maxCount: 1)
        let image = data.all.items[0].image

        let cropped = image.croppingContents()
        let centered = cropped.addingBlankSpace(toCenterInWidth: image.width, height: image.height, withXOffset: 0, yOffset: 0)

        XCTAssertEqual(centered.pixels.count, image.pixels.count)
        XCTAssertEqual(centered.width, image.width)
        XCTAssertEqual(centered.height, image.height)

        let diff = centered.pixels.difference(from: image.pixels)

        XCTAssertEqual(diff.count, 0)

        let reCropped = centered.croppingContents()
        let offsetCentered = reCropped.addingBlankSpace(toCenterInWidth: image.width + 100, height: image.height + 100, withXOffset: 40, yOffset: 50)
        let croppedAgain = offsetCentered.croppingContents()
        let centeredOriginalSize = croppedAgain.addingBlankSpace(toCenterInWidth: image.width, height: image.height, withXOffset: 0, yOffset: 0)

        let diff2 = centeredOriginalSize.pixels.difference(from: image.pixels)

        XCTAssertEqual(diff2.count, 0)
    }

    func testScaling() async {
        let data = await DataLoading.loadTrainingData(maxCount: 1)
        let image = data.all.items[0].image

        let scaled = image.downscaling(byFactor: 0.5)

        XCTAssertEqual(scaled.width, 14)
        XCTAssertEqual(scaled.height, 14)
        XCTAssertEqual(scaled.pixels.count, image.pixels.count / 4)
    }

    func testRotation() async {
        let data = await DataLoading.loadTrainingData(maxCount: 1)
        let image = data.all.items[0].image

        let rotated = image.rotating(by: .degrees(90))
        let rotatedBack = rotated.rotating(by: .degrees(-45))

        XCTAssertNotEqual(rotated.pixels, image.pixels)
        XCTAssertEqual(rotatedBack.pixels.count, image.pixels.count)

        // This doesn't pass, although the images do look the same
        XCTAssertEqual(zip(rotatedBack.pixels, image.pixels).numberOfElements(matching: { lhs, rhs in
            return abs(Int(lhs) - Int(rhs)) > 25
        }), 0)
    }
}
