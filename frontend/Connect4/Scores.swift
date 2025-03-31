//
//  Scores.swift
//  Connect4
//
//  Created by 江逸帆 on 3/31/25.
//

import SwiftUI

struct Scores: View {
    @State private var userScore: Int = 0
    @State private var aiScore: Int = 0
    
    var body: some View {
        HStack(spacing: 20) {
            // User score
            VStack {
                Text("You")
                    .font(.headline)
                Text("\(userScore)")
                    .font(.largeTitle)
            }
            .frame(width: 100, height: 100)
            .background(Color.blue.opacity(0.3))
            .cornerRadius(25)
            
            // VS
            VStack {
                Text("VS")
                    .font(.headline)
                    .foregroundColor(.gray)
            }
            .frame(width: 50, height: 50)
            
            // AI sciore
            VStack {
                Text("AI")
                    .font(.headline)
                Text("\(aiScore)")
                    .font(.largeTitle)
            }
            .frame(width: 100, height: 100)
            .background(Color.red.opacity(0.3))
            .cornerRadius(25)
        }
        .padding()
    }
}

#Preview {
    Scores()
}
