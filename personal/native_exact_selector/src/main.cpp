#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

struct Dataset {
    uint32_t samples{}, features{}, classes{}, groups{};
    std::vector<float> pairs, x;
    std::vector<int32_t> y, group;
};

template <class T> void read_exact(std::ifstream& in, std::vector<T>& values, size_t count) {
    values.resize(count);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) throw std::runtime_error("Truncated dataset file");
}

Dataset load_dataset(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open dataset: " + path);
    char magic[8]; in.read(magic, 8);
    if (std::string(magic, 8) != "HSIEXACT") throw std::runtime_error("Invalid dataset magic");
    Dataset d;
    in.read(reinterpret_cast<char*>(&d.samples), 4); in.read(reinterpret_cast<char*>(&d.features), 4);
    in.read(reinterpret_cast<char*>(&d.classes), 4); in.read(reinterpret_cast<char*>(&d.groups), 4);
    read_exact(in, d.pairs, static_cast<size_t>(d.features) * 2);
    read_exact(in, d.y, d.samples); read_exact(in, d.group, d.samples);
    read_exact(in, d.x, static_cast<size_t>(d.samples) * d.features);
    return d;
}

std::vector<int> make_folds(const Dataset& d, int folds, uint32_t seed) {
    std::vector<int> group_class(d.groups, -1), assignment(d.groups, -1);
    for (uint32_t s = 0; s < d.samples; ++s) group_class[d.group[s]] = d.y[s];
    std::mt19937 rng(seed);
    for (uint32_t c = 0; c < d.classes; ++c) {
        std::vector<int> ids;
        for (uint32_t g = 0; g < d.groups; ++g) if (group_class[g] == static_cast<int>(c)) ids.push_back(g);
        if (static_cast<int>(ids.size()) < folds) throw std::runtime_error("More folds than objects in a class");
        std::shuffle(ids.begin(), ids.end(), rng);
        for (size_t i = 0; i < ids.size(); ++i) assignment[ids[i]] = static_cast<int>(i % folds);
    }
    std::vector<int> sample_fold(d.samples);
    for (uint32_t s = 0; s < d.samples; ++s) sample_fold[s] = assignment[d.group[s]];
    return sample_fold;
}

struct FoldData { std::vector<int> samples; std::vector<float> contribution; };

std::vector<FoldData> precompute(const Dataset& d, const std::vector<int>& sample_fold, int folds) {
    std::vector<FoldData> output(folds);
    for (int fold = 0; fold < folds; ++fold) {
        auto& out = output[fold];
        for (uint32_t s = 0; s < d.samples; ++s) if (sample_fold[s] == fold) out.samples.push_back(s);
        const size_t stride = static_cast<size_t>(out.samples.size()) * d.classes;
        out.contribution.resize(static_cast<size_t>(d.features) * stride);
        std::vector<double> sum(static_cast<size_t>(d.features) * d.classes), square(sum.size());
        std::vector<int> count(d.classes);
        for (uint32_t s = 0; s < d.samples; ++s) if (sample_fold[s] != fold) {
            const int cls = d.y[s]; ++count[cls];
            for (uint32_t f = 0; f < d.features; ++f) {
                const double value = d.x[static_cast<size_t>(s) * d.features + f];
                const size_t pos = static_cast<size_t>(f) * d.classes + cls;
                sum[pos] += value; square[pos] += value * value;
            }
        }
        for (uint32_t f = 0; f < d.features; ++f) {
            double pooled_variance = 0.0; int pooled_count = 0;
            for (uint32_t c = 0; c < d.classes; ++c) {
                const size_t p = static_cast<size_t>(f) * d.classes + c;
                const double mean = sum[p] / count[c];
                pooled_variance += square[p] - count[c] * mean * mean; pooled_count += count[c] - 1;
            }
            const double variance = std::max(pooled_variance / std::max(1, pooled_count), 1e-12);
            for (size_t local = 0; local < out.samples.size(); ++local) {
                const float value = d.x[static_cast<size_t>(out.samples[local]) * d.features + f];
                for (uint32_t c = 0; c < d.classes; ++c) {
                    const double mean = sum[static_cast<size_t>(f) * d.classes + c] / count[c];
                    const double delta = value - mean;
                    out.contribution[static_cast<size_t>(f) * stride + local * d.classes + c] = static_cast<float>(delta * delta / variance);
                }
            }
        }
    }
    return output;
}

struct Result { float score; uint16_t a, b, c; };
struct Worse { bool operator()(const Result& x, const Result& y) const { return x.score > y.score; } };

float evaluate(const Dataset& d, const std::vector<FoldData>& folds, int a, int b, int c) {
    float total = 0.0f;
    for (const auto& fold : folds) {
        const size_t stride = fold.samples.size() * d.classes;
        const float *pa = fold.contribution.data() + static_cast<size_t>(a) * stride;
        const float *pb = fold.contribution.data() + static_cast<size_t>(b) * stride;
        const float *pc = fold.contribution.data() + static_cast<size_t>(c) * stride;
        std::vector<int> correct(d.classes), count(d.classes);
        for (size_t s = 0; s < fold.samples.size(); ++s) {
            int predicted = 0; float best = pa[s*d.classes] + pb[s*d.classes] + pc[s*d.classes];
            for (uint32_t cls = 1; cls < d.classes; ++cls) {
                const size_t p = s*d.classes + cls; const float distance = pa[p] + pb[p] + pc[p];
                if (distance < best) { best = distance; predicted = static_cast<int>(cls); }
            }
            const int truth = d.y[fold.samples[s]]; ++count[truth]; if (predicted == truth) ++correct[truth];
        }
        float balanced = 0.0f;
        for (uint32_t cls = 0; cls < d.classes; ++cls) balanced += static_cast<float>(correct[cls]) / count[cls];
        total += balanced / d.classes;
    }
    return total / folds.size();
}

float evaluate_subset(const Dataset& d, const std::vector<FoldData>& folds, const std::vector<uint16_t>& subset) {
    float total = 0.0f;
    for (const auto& fold : folds) {
        const size_t stride = fold.samples.size() * d.classes;
        std::vector<int> correct(d.classes), count(d.classes);
        for (size_t s = 0; s < fold.samples.size(); ++s) {
            int predicted = 0; float best = 0.0f;
            for (uint16_t f : subset) best += fold.contribution[static_cast<size_t>(f)*stride+s*d.classes];
            for (uint32_t cls=1; cls<d.classes; ++cls) {
                float distance=0.0f;
                for (uint16_t f : subset) distance += fold.contribution[static_cast<size_t>(f)*stride+s*d.classes+cls];
                if (distance < best) { best=distance; predicted=static_cast<int>(cls); }
            }
            const int truth=d.y[fold.samples[s]]; ++count[truth]; if(predicted==truth) ++correct[truth];
        }
        float balanced=0.0f;
        for(uint32_t cls=0;cls<d.classes;++cls) balanced += static_cast<float>(correct[cls])/count[cls];
        total += balanced/d.classes;
    }
    return total/folds.size();
}

struct Candidate { float score{}; std::vector<uint16_t> indices; };

std::string subset_key(const std::vector<uint16_t>& subset) {
    std::string key(subset.size()*sizeof(uint16_t), '\0');
    std::memcpy(key.data(), subset.data(), key.size());
    return key;
}

std::vector<Candidate> beam_search(const Dataset& d, const std::vector<FoldData>& folds, int target, int beam_width, unsigned threads) {
    std::vector<Candidate> beam(1);
    for(int depth=0;depth<target;++depth) {
        std::vector<Candidate> candidates; std::unordered_set<std::string> seen;
        candidates.reserve(beam.size()*(d.features-depth));
        for(const auto& parent:beam) for(uint16_t feature=0;feature<d.features;++feature) {
            if(std::binary_search(parent.indices.begin(),parent.indices.end(),feature)) continue;
            auto indices=parent.indices;
            indices.insert(std::lower_bound(indices.begin(),indices.end(),feature),feature);
            if(seen.insert(subset_key(indices)).second) candidates.push_back({0.0f,std::move(indices)});
        }
        std::atomic<size_t> next{0};
        auto worker=[&]{while(true){size_t i=next.fetch_add(1);if(i>=candidates.size())break;candidates[i].score=evaluate_subset(d,folds,candidates[i].indices);}};
        std::vector<std::thread> pool; for(unsigned i=0;i<threads;++i)pool.emplace_back(worker);for(auto&t:pool)t.join();
        const size_t tested=candidates.size();
        const size_t keep=std::min(static_cast<size_t>(beam_width),tested);
        std::partial_sort(candidates.begin(),candidates.begin()+keep,candidates.end(),[](const Candidate&a,const Candidate&b){return a.score>b.score;});
        candidates.resize(keep);beam=std::move(candidates);
        std::cout<<"Selected "<<depth+1<<'/'<<target<<" bands; tested "<<tested<<" unique subsets; best score "<<beam.front().score<<'\n';
    }
    return beam;
}

Candidate swap_refine(const Dataset& d,const std::vector<FoldData>& folds,Candidate best,unsigned threads){
    for(int pass=1;pass<=10;++pass){
        std::vector<Candidate> trials;
        for(size_t position=0;position<best.indices.size();++position)for(uint16_t feature=0;feature<d.features;++feature){
            if(std::binary_search(best.indices.begin(),best.indices.end(),feature))continue;
            auto indices=best.indices;indices[position]=feature;std::sort(indices.begin(),indices.end());trials.push_back({0.0f,std::move(indices)});
        }
        std::atomic<size_t> next{0};auto worker=[&]{while(true){size_t i=next.fetch_add(1);if(i>=trials.size())break;trials[i].score=evaluate_subset(d,folds,trials[i].indices);}};
        std::vector<std::thread> pool;for(unsigned i=0;i<threads;++i)pool.emplace_back(worker);for(auto&t:pool)t.join();
        auto improved=std::max_element(trials.begin(),trials.end(),[](const Candidate&a,const Candidate&b){return a.score<b.score;});
        if(improved==trials.end()||improved->score<=best.score+1e-7f)break;
        best=*improved;std::cout<<"Swap refinement pass "<<pass<<"; score "<<best.score<<'\n';
    }
    return best;
}

int main(int argc, char** argv) try {
    if (argc < 2) { std::cerr << "Usage: lichen_exact DATASET [OUTPUT.csv] [N=13] [THREADS] [BEAM=128] [TOP_K=100]\n"; return 2; }
    const std::string output = argc > 2 ? argv[2] : "results/top_triples.csv";
    const int target = argc > 3 ? std::stoi(argv[3]) : 13;
    const unsigned threads = argc > 4 ? std::stoul(argv[4]) : std::max(1u, std::thread::hardware_concurrency());
    const int beam_width = argc > 5 ? std::stoi(argv[5]) : 128;
    const int top_k = argc > 6 ? std::stoi(argv[6]) : 100;
    const auto started = std::chrono::steady_clock::now();
    Dataset d = load_dataset(argv[1]);
    if(target<1||target>static_cast<int>(d.features))throw std::runtime_error("N must be between 1 and the feature count");
    auto folds = precompute(d, make_folds(d, 3, 42), 3);
    if(target!=3){
        std::cout<<"Native beam search: selecting "<<target<<" of "<<d.features<<" pairs, beam "<<beam_width<<", "<<threads<<" threads\n";
        auto results=beam_search(d,folds,target,beam_width,threads);
        Candidate refined=swap_refine(d,folds,results.front(),threads);
        if(refined.score>results.front().score)results.insert(results.begin(),refined);
        std::sort(results.begin(),results.end(),[](const Candidate&a,const Candidate&b){return a.score>b.score;});
        std::ofstream csv(output);if(!csv)throw std::runtime_error("Cannot create output: "+output);
        csv<<"rank,score";for(int i=0;i<target;++i)csv<<",index_"<<i+1<<",ex_"<<i+1<<",em_"<<i+1;csv<<'\n';
        const size_t limit=std::min(static_cast<size_t>(top_k),results.size());
        for(size_t rank=0;rank<limit;++rank){const auto&r=results[rank];csv<<rank+1<<','<<std::setprecision(9)<<r.score;for(auto f:r.indices)csv<<','<<f<<','<<d.pairs[2*f]<<','<<d.pairs[2*f+1];csv<<'\n';}
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cout<<"Best balanced accuracy: "<<results.front().score<<"\nSaved: "<<output<<"\nElapsed: "<<seconds<<" seconds\n";return 0;
    }
    const uint64_t combinations = static_cast<uint64_t>(d.features) * (d.features-1) * (d.features-2) / 6;
    std::cout << "Exact search: " << combinations << " triples, " << threads << " threads\n";
    std::atomic<int> next_a{0}; std::mutex result_mutex;
    std::priority_queue<Result, std::vector<Result>, Worse> global;
    auto worker = [&] {
        std::priority_queue<Result, std::vector<Result>, Worse> local;
        while (true) {
            const int a = next_a.fetch_add(1); if (a >= static_cast<int>(d.features)-2) break;
            for (int b = a+1; b < static_cast<int>(d.features)-1; ++b) for (int c = b+1; c < static_cast<int>(d.features); ++c) {
                Result r{evaluate(d, folds, a, b, c), static_cast<uint16_t>(a), static_cast<uint16_t>(b), static_cast<uint16_t>(c)};
                if (static_cast<int>(local.size()) < top_k) local.push(r); else if (r.score > local.top().score) { local.pop(); local.push(r); }
            }
        }
        std::lock_guard lock(result_mutex);
        while (!local.empty()) { auto r=local.top(); local.pop(); if (static_cast<int>(global.size()) < top_k) global.push(r); else if (r.score > global.top().score) { global.pop(); global.push(r); } }
    };
    std::vector<std::thread> pool; for (unsigned i=0; i<threads; ++i) pool.emplace_back(worker); for (auto& t:pool) t.join();
    std::vector<Result> results; while (!global.empty()) { results.push_back(global.top()); global.pop(); } std::reverse(results.begin(), results.end());
    std::ofstream csv(output); if (!csv) throw std::runtime_error("Cannot create output; create its parent directory first: " + output);
    csv << "rank,score,index_a,ex_a,em_a,index_b,ex_b,em_b,index_c,ex_c,em_c\n";
    for (size_t rank=0; rank<results.size(); ++rank) { const auto&r=results[rank]; csv << rank+1 << ',' << std::setprecision(9) << r.score << ',' << r.a << ',' << d.pairs[2*r.a] << ',' << d.pairs[2*r.a+1] << ',' << r.b << ',' << d.pairs[2*r.b] << ',' << d.pairs[2*r.b+1] << ',' << r.c << ',' << d.pairs[2*r.c] << ',' << d.pairs[2*r.c+1] << '\n'; }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::cout << "Best balanced accuracy: " << results.front().score << "\nSaved: " << output << "\nElapsed: " << seconds << " seconds\n";
    return 0;
} catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
