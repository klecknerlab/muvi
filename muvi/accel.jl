function decompress_blocks(input::AbstractVector{UInt8}, block_size::Integer, last_block_size::Integer, block_ends::AbstractVector{UInt64}, output::AbstractVector{UInt8})
    num_blocks = length(block_ends)

    @Threads.threads for p in 1:num_blocks

        # Note that Julia is 1 indexed, so we need to add 1 a bunch of places!
        i = (p == 1) ? 1 : block_ends[p - 1] + 1
        block_end = block_ends[p]
        j = Int(block_size * (p-1)) + 1

        output_end = j + ((p == num_blocks) ? last_block_size : block_size) - 1

        # Note that the minimum token is two bytes, so if i == block_end we are done!
        # There might be one left in the output though...
        while (j <= output_end) && (i < block_end)
            t1 = Int(input[i] & 0xF0) >> 4
            t2 = Int(input[i] & 0x0F) + 4
            i += 1

            if (t1 == 15)
                while input[i] == 255
                    t1 += 255
                    i += 1
                end

                t1 += Int(input[i])
                i += 1
            end

            for n in 1:t1
                output[j] = input[i]
                i += 1
                j += 1
            end

            # The last block doesn't have a match -- lets check if we are there
            if (j > output_end) || (i > block_end)
                break
            end

            off = Int(input[i]) + (Int(input[i+1]) << 8)
            i += 2

            if (t2 == 19)
                while input[i] == 255
                    t2 += 255
                    i += 1
                end

                t2 += Int(input[i])
                i += 1
            end

            for n in 1:t2
                output[j] = output[j - off]
                j += 1
            end

        end
    end
end


function unpack_10b(input::AbstractVector{UInt8}, tone_map::AbstractVector{T}, output::AbstractVector{T}, offset::Integer=0, stride::Integer=1) where {T <: Integer}
    # 10 bits data => 4 points packed into 5 bytes
    @Threads.threads for block in 1:(length(input) ÷ 5)
        i = (block - 1) * 5 + 1
        val0 = (UInt16(input[i+0] & 0xFF) << 2) + (UInt16(input[i+1]) >> 6)
        val1 = (UInt16(input[i+1] & 0x3F) << 4) + (UInt16(input[i+2]) >> 4)
        val2 = (UInt16(input[i+2] & 0x0F) << 6) + (UInt16(input[i+3]) >> 2)
        val3 = (UInt16(input[i+3] & 0x03) << 8) + (UInt16(input[i+4]) >> 0)

        j = offset + ((block-1) * 4 * stride) + 1
        output[j           ] = tone_map[val0+1]
        output[j +   stride] = tone_map[val1+1]
        output[j + 2*stride] = tone_map[val2+1]
        output[j + 3*stride] = tone_map[val3+1]
    end
end