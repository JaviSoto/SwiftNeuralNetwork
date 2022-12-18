//
//  SampleImageView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI
import SwiftMatrix

struct PixelArrayImageView: View {
    let pixels: [ColorPixel]
    let width: UInt32
    let lazy: Bool

    @State
    private var cachedImage: Image?

    var image: Image? {
        if lazy {
            return cachedImage
        } else {
            return loadImage()
        }
    }

    init(pixels: [ColorPixel], width: UInt32, lazy: Bool = true) {
        self.pixels = pixels
        self.width = width
        self.lazy = lazy
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
        return pixels.asSwiftUIImage(width: width)
    }
}

extension PixelArrayImageView {
    init(sampleImage: SampleImage, width: UInt32, lazy: Bool = true) {
        self.init(pixels: sampleImage.pixels.map { $0.colorPixel }, width: width, lazy: lazy)
    }
}
