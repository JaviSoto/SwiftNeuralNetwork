//
//  SampleImageView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct SampleImageView: View {
    let sampleImage: SampleImage
    let width: UInt32

    var body: some View {
        sampleImage.asSwiftUIImage(width: width)
            .resizable()
            .aspectRatio(1, contentMode: .fit)
    }
}
