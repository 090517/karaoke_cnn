from helper.dataset_creation import build_training_data, build_pre_training_data
import global_settings as gs

print("-- Building pre-train data --")
build_pre_training_data(gs.vocal_path,
                        gs.pre_training_path,
                        file_amount=gs.pre_file_amount, 
                        intervals=gs.intervals,
                        chunk_size=gs.pre_chunk_size,  
                        snippets_per_file=gs.pre_snippets_per_file, 
                        n_fft = gs.n_fft,
                        index_offset = gs.index_offset,
                        sample_rate = gs.sample_rate,
                        hop_factor = gs.hop_factor,
                        puffer = gs.puffer
                        )

print("-- Building training data --")
build_training_data(gs.vocal_path,
                    gs.music_path,
                    gs.training_path,
                    file_amount=gs.file_amount,
                    intervals=gs.intervals,
                    chunk_size = gs.chunk_size,
                    snippets_per_file = gs.snippets_per_file,
                    music_only = gs.music_only,
                    n_fft = gs.n_fft,
                    index_offset = gs.index_offset,
                    sample_rate = gs.sample_rate,
                    hop_factor = gs.hop_factor,
                    puffer = gs.puffer
                    )

print("-- Data sets built --")
