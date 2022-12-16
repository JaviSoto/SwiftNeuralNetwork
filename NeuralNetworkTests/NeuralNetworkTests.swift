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
}
