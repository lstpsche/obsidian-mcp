//! Tantivy BM25 full-text index: schema definition and field handles.

use tantivy::schema::{
    Field, IndexRecordOption, STORED, STRING, Schema, SchemaBuilder, TextFieldIndexing, TextOptions,
};

pub const TITLE_BOOST: f32 = 5.0;
pub const HEADINGS_BOOST: f32 = 3.0;
pub const TAGS_BOOST: f32 = 4.0;
pub const FRONTMATTER_BOOST: f32 = 2.0;

/// Pre-built Tantivy schema with cached field handles for fast document construction.
pub struct TantivySchema {
    pub schema: Schema,
    /// Vault-relative path (stored, not indexed — used as primary key).
    pub f_path: Field,
    /// Note title / filename stem (stored, indexed with `en_stem`, high boost).
    pub f_title: Field,
    /// Concatenated headings (not stored, indexed with `en_stem`, medium boost).
    pub f_headings: Field,
    /// Tags as facets (stored, indexed).
    pub f_tags: Field,
    /// Full note body (not stored, indexed with `en_stem`, base boost).
    pub f_body: Field,
    /// Stringified frontmatter values (not stored, indexed with `en_stem`).
    pub f_frontmatter_text: Field,
}

impl TantivySchema {
    /// Build the Tantivy schema for vault note indexing.
    ///
    /// All indexed text fields use the built-in `en_stem` tokenizer (English stemmer)
    /// with `WithFreqsAndPositions` to support phrase queries and proximity scoring.
    pub fn build() -> Self {
        let mut builder = SchemaBuilder::new();

        let f_path = builder.add_text_field("path", STRING | STORED);

        let stemmed_indexing = TextFieldIndexing::default()
            .set_tokenizer("en_stem")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);

        let f_title = builder.add_text_field(
            "title",
            TextOptions::default()
                .set_indexing_options(stemmed_indexing.clone())
                .set_stored(),
        );

        let f_headings = builder.add_text_field(
            "headings",
            TextOptions::default().set_indexing_options(stemmed_indexing.clone()),
        );

        let f_tags = builder.add_facet_field("tags", STORED);

        let f_body = builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(stemmed_indexing.clone()),
        );

        let f_frontmatter_text = builder.add_text_field(
            "frontmatter_text",
            TextOptions::default().set_indexing_options(stemmed_indexing),
        );

        let schema = builder.build();

        Self {
            schema,
            f_path,
            f_title,
            f_headings,
            f_tags,
            f_body,
            f_frontmatter_text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_expected_fields() {
        let ts = TantivySchema::build();

        assert_eq!(ts.schema.get_field_name(ts.f_path), "path");
        assert_eq!(ts.schema.get_field_name(ts.f_title), "title");
        assert_eq!(ts.schema.get_field_name(ts.f_headings), "headings");
        assert_eq!(ts.schema.get_field_name(ts.f_tags), "tags");
        assert_eq!(ts.schema.get_field_name(ts.f_body), "body");
        assert_eq!(
            ts.schema.get_field_name(ts.f_frontmatter_text),
            "frontmatter_text"
        );
    }

    #[test]
    fn path_field_is_stored_and_string_indexed() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_path);

        assert!(entry.is_stored());
        assert!(entry.is_indexed());
    }

    #[test]
    fn title_field_is_stored_and_stemmed() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_title);

        assert!(entry.is_stored());
        assert!(entry.is_indexed());
    }

    #[test]
    fn body_field_is_not_stored() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_body);

        assert!(!entry.is_stored());
        assert!(entry.is_indexed());
    }

    #[test]
    fn headings_field_is_not_stored() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_headings);

        assert!(!entry.is_stored());
        assert!(entry.is_indexed());
    }

    #[test]
    fn tags_field_is_facet_and_stored() {
        let ts = TantivySchema::build();
        let entry = ts.schema.get_field_entry(ts.f_tags);

        assert!(entry.is_stored());
    }

    #[test]
    fn schema_field_count() {
        let ts = TantivySchema::build();

        assert_eq!(ts.schema.num_fields(), 6);
    }
}
