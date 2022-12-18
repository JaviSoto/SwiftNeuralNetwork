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
    let lazy: Bool

    init(sampleImage: SampleImage, width: UInt32, lazy: Bool = true) {
        self.sampleImage = sampleImage
        self.width = width
        self.lazy = lazy
    }

    @State
    private var cachedImage: Image?

    var image: Image? {
        if lazy {
            return cachedImage
        } else {
            return loadImage()
        }
    }

    var body: some View {
        Group {
            if let image {
                image
                    .resizable()
                    .aspectRatio(1, contentMode: .fit)
            }
        }
        .onAppear {
            if lazy {
                cachedImage = loadImage()
            }
        }
    }

    private func loadImage() -> Image {
        return sampleImage.asSwiftUIImage(width: width)
    }
}
