"""Utility to regenerate benchmark fixtures (texts and tokenizer JSON files)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordLevel, WordPiece
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer


def build_text_fixtures(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    news_text = dedent(
        """
        City officials confirmed on Tuesday that the riverside park will reopen this summer after a two-year renovation.
        Crews installed 175 energy-efficient lights, replanted native wildflowers, and added a playground designed by local artists.
        The project ran $1.8 million under budget, according to Deputy Mayor Alicia Gómez — a welcome surprise for residents concerned about rising taxes.
        "It's not just a facelift; it's a commitment to public space," said Gomez. Cyclists tested the new bike lanes, while children chased bubbles during the ribbon-cutting ceremony.
        The park will host weekly night markets featuring Afghan bolani, Jamaican patties, and vegan empanadas, with vendors selected through a community ballot.
        Public transit advocates noted that the expanded bus schedule, combined with real-time arrival boards, should alleviate weekend congestion.
        Sustainability officers also unveiled a solar-powered irrigation system and a pollinator habitat that includes milkweed, lavender, and rare prairie clover.
        Early visitor surveys show 92% satisfaction, with many praising the accessible design, tactile maps, and multilingual audio tours available in English, Spanish, Mandarin, and American Sign Language.
        The city plans to share open-source blueprints and a detailed maintenance playbook with other municipalities considering similar upgrades.
        """
    ).strip()

    wiki_base = dedent(
        """
        == Early History ==

        The settlement traces its roots to a trading village documented in the 12th-century annals of the Seljuk chronicler al-Biruni. Archaeological digs in 1989 uncovered kiln-fired ceramics, copper ingots, and terraced irrigation canals that reshaped historians' understanding of Central Asian trade routes.
        == Linguistics ==
        Modern dialect surveys reveal a blend of Chuvash, Khazar, and Oghur loanwords; linguists have mapped palatalized consonants appearing near river valleys, likely a relic of seasonal migration.
        == Technological Renaissance ==
        By 1893 the town hosted one of the earliest wireless telegraph stations in the region. Engineer Lidiya Petrovna retrofitted surplus naval equipment to send meteorological data to Moscow every sunset. Her notebooks — digitized in 2017 — contain meticulous diagrams of spark-gap transmitters, annotations in French, and the occasional doodle of a cat wearing goggles.
        == Cultural Revival ==
        Annual festivals now feature Tuvan throat singing workshops, VR reconstructions of vanished monasteries, and fermentation labs explaining the chemistry behind kumis. UNESCO added the town's accordion workshops to its intangible heritage list, citing their adaptive use of recycled polymers for reeds.
        == Contemporary Research ==
        In 2022 a consortium of botanists, data journalists, and Indigenous seed keepers launched the Steppe Observatory, using open satellite data, LoRaWAN sensors, and community weather diaries to forecast dust storms.
        == Notable Figures ==
        Historian Salome Okafor popularized the settlement after translating 400 folktales into Yoruba, English, and Esperanto, each annotated with QR codes linking to oral history recordings.
        == Gastronomy ==
        Local chefs pair fermented camel-milk cheese with candied sea buckthorn, while food trucks experiment with kelp-laden naan tacos, reflecting the town's fishing diaspora.
        == Climate Adaptation ==
        Flood mitigation now involves mycelium-reinforced levees, willow microforests, and AI-optimized sluice gates governed by a civic algorithm crafted in nightly town halls.
        == Digital Archives ==
        Volunteer coders maintain a mirrored archive stored on solar-powered Raspberry Pi clusters. The archive syncs monthly via a community-owned satellite uplink leased during lunar downtimes.
        == Everyday Life ==
        Schoolchildren log phenology observations, while retired tram conductors teach visitor orientation classes in a repurposed depot, complete with time-travel escape room puzzles chronicling the town's evolution.
        """
    ).strip()

    code_excerpt = dedent(
        """
        // telemetry.rs — data ingestion service
        use chrono::{DateTime, Utc};
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Serialize, Deserialize)]
        pub struct Reading {
            pub station_id: String,
            pub recorded_at: DateTime<Utc>,
            pub metrics: Vec<f32>,
        }

        impl Reading {
            pub fn normalize(&mut self) {
                let max = self.metrics.iter().cloned().fold(f32::NAN, f32::max);
                if max.is_finite() && max > 0.0 {
                    for value in &mut self.metrics {
                        *value /= max;
                    }
                }
            }

            pub fn annotate(&self) -> String {
                format!("station: {} | samples: {}", self.station_id, self.metrics.len())
            }
        }

        #[tokio::main]
        async fn main() -> anyhow::Result<()> {
            let mut reader = telemetry::stream("wss://edge.router.local:7443").await?;
            while let Some(mut reading) = reader.next().await {
                reading.normalize();
                tracing::info!(%reading.station_id, %reading.metrics.len(), "received reading");
                storage::persist(reading).await?;
            }
            Ok(())
        }
        """
    ).strip()

    wiki_text = (wiki_base + "\n\n") * 9
    while len(wiki_text) < 64 * 1024:
        wiki_text += wiki_base + "\n\n"
    wiki_text = wiki_text[: 64 * 1024]

    (data_dir / "news_1k.txt").write_text(news_text, encoding="utf-8")
    (data_dir / "wiki_64k.txt").write_text(wiki_text, encoding="utf-8")
    (data_dir / "code_excerpt.txt").write_text(code_excerpt, encoding="utf-8")


def train_tokenizers(data_dir: Path) -> None:
    corpus = [
        (data_dir / "news_1k.txt").read_text(encoding="utf-8"),
        (data_dir / "wiki_64k.txt").read_text(encoding="utf-8"),
        (data_dir / "code_excerpt.txt").read_text(encoding="utf-8"),
    ]

    bpe_tokenizer = Tokenizer(BPE(unk_token="<|unk|>", fuse_unk=True))
    bpe_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    bpe_tokenizer.decoder = ByteLevelDecoder()
    bpe_trainer = BpeTrainer(
        vocab_size=2000,
        min_frequency=2,
        special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"],
    )
    bpe_tokenizer.train_from_iterator(corpus, bpe_trainer)
    bpe_tokenizer.save(str(data_dir / "byte_bpe.json"))

    wp_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]", continuing_subword_prefix="##"))
    wp_tokenizer.normalizer = Lowercase()
    wp_tokenizer.pre_tokenizer = Whitespace()
    wp_trainer = WordPieceTrainer(
        vocab_size=2000,
        min_frequency=2,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
    )
    wp_tokenizer.train_from_iterator(corpus, wp_trainer)
    wp_tokenizer.save(str(data_dir / "wordpiece.json"))

    wl_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    wl_tokenizer.normalizer = Lowercase()
    wl_tokenizer.pre_tokenizer = Whitespace()
    wl_trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]"])
    wl_tokenizer.train_from_iterator(corpus, wl_trainer)
    wl_tokenizer.save(str(data_dir / "wordlevel.json"))


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    build_text_fixtures(data_dir)
    train_tokenizers(data_dir)
    print(f"Fixtures regenerated in {data_dir}")


if __name__ == "__main__":
    main()
