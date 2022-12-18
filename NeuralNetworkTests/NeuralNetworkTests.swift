//
//  NeuralNetworkTests.swift
//  NeuralNetworkTests
//
//  Created by Javier Soto on 12/14/22.
//

import XCTest
import SwiftMatrix
@testable import NeuralNetwork

final class NeuralNetworkTests: XCTestCase {
    func testMatrixMultiplication() {
        let matrixA = Matrix([[1, 2, 3], [4, 5, 6]])
        let matrixB = Matrix([[7, 8], [9, 10], [11, 12]])

        XCTAssertEqual(matrixA ° matrixB, Matrix([[58, 64], [139, 154]]))
    }

    func testMatrixAddition() {
        let matrix = Matrix([[1, 2, 3], [4, 5, 6]])

        XCTAssertEqual(matrix + 1, Matrix([[2, 3, 4], [5, 6, 7]]))
    }

    func testMatrixSum() {
        let matrix = Matrix([[1, 2, 3], [4, 5, 6]])

        XCTAssertEqual(matrix.sum(), 21)
    }

    func testMatrixSumMatrix() {
        let matrix = Matrix([[1, 2, 3], [4, 5, 6]])

        XCTAssertEqual(matrix.sumMatrix(), Matrix([[5, 7, 9]]))
    }

    func testOneHotMatrix() {
        let values: [Double] = [1, 9, 2, 0]
        let testOutputs = Matrix(rows: values.count, columns: 1, values: values)
        let expectedResult = Matrix([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])′

        let result = testOutputs.oneHot(withOutputLayerSize: 10)

        XCTAssertEqual(result.rows, 10)
        XCTAssertEqual(result.columns, values.count)

        XCTAssertEqual(result, expectedResult)
    }

    func testAccuracyCalculation() {
        let output = Matrix([
            [0.6, 0.3, 0.2, 0.1, 0.4, 0.12, 0.3, 0.25, 0.11, 0.05], // 0 is highest
            [0.1, 0.9, 0.2, 0.1, 0.4, 0.12, 0.3, 0.25, 0.11, 0.05], // 1 is highest
            [0.1, 0.7, 0.2, 0.1, 0.4, 0.12, 0.3, 0.25, 0.11, 0.85], // 9 is highest
            [0.1, 0.7, 0.2, 0.1, 0.4, 0.12, 0.3, 0.25, 0.11, 0.45], // 1 is highest
            [0.1, 0.7, 0.2, 0.1, 0.4, 0.12, 0.3, 0.25, 0.11, 0.45], // 1 is highest
            [0.1, 0.7, 0.2, 0.89, 0.4, 0.12, 0.3, 0.25, 0.11, 0.45], // 3 is highest
        ])′
        let validationData = Matrix([[
            0, // Correct
            2, // Incorrect
            9, // Correct
            4, // Incorrect
            1, // Correct
            6 // Incorrect
        ]])′

        XCTAssertEqual(NeuralNetwork.accuracy(ofOutput: output, againstValidationData: validationData), 0.5)
    }
}
