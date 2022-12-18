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
        let matrixA = Matrix([
            [1, 2, 3],
            [4, 5, 6]
        ])
        let matrixB = Matrix([
            [7, 8],
            [9, 10],
            [11, 12]
        ])

        let expectedResult = Matrix([
            [58, 64],
            [139, 154]
        ])

        XCTAssertEqual(matrixA ° matrixB, expectedResult)
    }

    func testMatrixAddition() {
        let matrix = Matrix([
            [1, 2, 3],
            [4, 5, 6]
        ])

        let expectedResult = Matrix([
            [2, 3, 4],
            [5, 6, 7]
        ])

        XCTAssertEqual(matrix + 1, expectedResult)
    }

    func testMatrixSum() {
        let matrix = Matrix([
            [1, 2, 3],
            [4, 5, 6]
        ])

        XCTAssertEqual(matrix.sum(), 21)
    }

    func testMatrixSumMatrix() {
        let matrix = Matrix(
            [[1, 2, 3],
             [4, 5, 6]
            ])

        XCTAssertEqual(matrix.sumMatrix(), Matrix([[5, 7, 9]]))
    }

    func testGettingMatrixRow() {
        let matrix = Matrix([
            [1, 2, 3],
            [4, 5, 6]
        ])

        XCTAssertEqual(matrix.row(0), [1, 2, 3])
        XCTAssertEqual(matrix.row(1), [4, 5, 6])
    }

    func testGettingMatrixColumn() {
        let matrix = Matrix([
            [1, 2, 3],
            [4, 5, 6]
        ])

        XCTAssertEqual(matrix.column(0), [1, 4])
        XCTAssertEqual(matrix.column(1), [2, 5])
        XCTAssertEqual(matrix.column(2), [3, 6])
    }

    func testGettingMatrixRows() {
        let matrix = Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        XCTAssertEqual(matrix.rows([0, 2]), Matrix([[1, 2, 3], [7, 8, 9]]))
        XCTAssertEqual(matrix.rows([1]), Matrix([[4, 5, 6]]))
    }

    func testGettingMatrixColumns() {
        let matrix = Matrix([
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [7, 8, 9, 10]
        ])

        XCTAssertEqual(matrix.columns([0, 2, 3]), Matrix([
            [1, 3, 4],
            [4, 6, 7],
            [7, 9, 10]
        ]))
        XCTAssertEqual(matrix.columns([1]), Matrix([
            [2],
            [5],
            [8]
        ]))
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
