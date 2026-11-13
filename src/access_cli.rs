//! Local owner administration and explicit client credential delivery.

use std::collections::{BTreeMap, HashSet};
use std::ffi::OsString;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use obsidian_mcp::access::clients::{
    AccessAction, AccessPolicy, ClientPolicy, Credential, DecisionReason, Deny, Grant, Role,
    SourceIdentity, validate_client_id,
};
use obsidian_mcp::access::store::{
    AccessError, AccessStore, default_data_dir, default_profile_path, read_token_file,
    validate_initialization, write_protected_file,
};
use obsidian_mcp::error::VaultError;
use obsidian_mcp::vault::fs::VaultRoot;
use serde_json::{Value, json};

const DEFAULT_URL: &str = "http://127.0.0.1:37842/mcp";
const MODERN: &str = "2026-07-28";
const MAX_PREVIEW: usize = 1000;
const MAX_HTTP_BYTES: usize = 1024 * 1024;

/// Arguments start after the `access` subcommand.
pub async fn run(args: Vec<OsString>) -> i32 {
    let args = match Arguments::parse(args) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error}\nUse 'obsidian-mcp access --help' for usage.");
            return 2;
        }
    };
    if args.flag("help") || args.words.is_empty() {
        print_help();
        return 0;
    }
    match execute(&args).await {
        Ok(output) => {
            if args.flag("json") || output.protocol {
                match serde_json::to_string_pretty(&output.value) {
                    Ok(value) => println!("{value}"),
                    Err(_) => {
                        eprintln!("error: could not encode command result");
                        return 1;
                    }
                }
            } else {
                for line in output.lines {
                    println!("{line}");
                }
            }
            0
        }
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}

struct Output {
    value: Value,
    lines: Vec<String>,
    protocol: bool,
}

impl Output {
    fn new(value: Value, lines: impl IntoIterator<Item = String>) -> Self {
        Self {
            value,
            lines: lines.into_iter().collect(),
            protocol: false,
        }
    }
}

#[derive(Default)]
struct Arguments {
    words: Vec<OsString>,
    values: BTreeMap<String, Vec<OsString>>,
    flags: HashSet<String>,
}

impl Arguments {
    fn parse(values: Vec<OsString>) -> Result<Self, String> {
        let mut args = Self::default();
        let mut values = values.into_iter();
        while let Some(value) = values.next() {
            let text = value.to_str();
            if text == Some("--") {
                args.words.extend(values);
                break;
            }
            if matches!(text, Some("--help" | "-h")) {
                args.flags.insert("help".into());
                continue;
            }
            if let Some(option) = text.and_then(|text| text.strip_prefix("--")) {
                match option {
                    "json" | "dry-run" | "semantic" => {
                        if !args.flags.insert(option.into()) {
                            return Err(format!("option --{option} was supplied more than once"));
                        }
                    }
                    "access-file" | "vault" | "data-dir" | "directory" | "file" | "role"
                    | "deny-directory" | "deny-file" | "token-out" | "out" | "expires-at"
                    | "expected-revision" | "action" | "path" | "url" | "client" | "token-file"
                    | "client-token-file" | "binary" | "name" | "max-results" | "write-check"
                    | "transport" => {
                        let value = values
                            .next()
                            .ok_or_else(|| format!("missing value for --{option}"))?;
                        if value.as_encoded_bytes().starts_with(b"--") {
                            return Err(format!("missing value for --{option}"));
                        }
                        args.values.entry(option.into()).or_default().push(value);
                    }
                    _ => return Err(format!("unknown option --{option}")),
                }
            } else if value.as_encoded_bytes().starts_with(b"-") {
                return Err(format!("unknown option {}", value.to_string_lossy()));
            } else {
                args.words.push(value);
            }
        }
        for (name, values) in &args.values {
            if values.len() > 1
                && !matches!(
                    name.as_str(),
                    "directory" | "file" | "deny-directory" | "deny-file"
                )
            {
                return Err(format!("option --{name} was supplied more than once"));
            }
        }
        Ok(args)
    }

    fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }

    fn value(&self, key: &str) -> Option<&OsString> {
        self.values.get(key).and_then(|values| values.first())
    }

    fn text(&self, key: &str) -> Result<Option<&str>, String> {
        self.value(key)
            .map(|value| {
                value
                    .to_str()
                    .ok_or_else(|| format!("--{key} must be valid UTF-8"))
            })
            .transpose()
    }

    fn required_text(&self, key: &str) -> Result<&str, String> {
        self.text(key)?
            .ok_or_else(|| format!("missing --{key}; use 'obsidian-mcp access --help'"))
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
        self.value(key).map(PathBuf::from)
    }

    fn word(&self, index: usize) -> Result<&str, String> {
        self.words
            .get(index)
            .and_then(|word| word.to_str())
            .ok_or_else(|| "missing command argument; use 'obsidian-mcp access --help'".into())
    }

    fn require_words(&self, count: usize) -> Result<(), String> {
        if self.words.len() != count {
            return Err(
                "unexpected or missing command argument; use 'obsidian-mcp access --help'".into(),
            );
        }
        Ok(())
    }

    fn revision(&self) -> Result<Option<u64>, String> {
        self.text("expected-revision")?
            .map(|value| {
                value
                    .parse::<u64>()
                    .map_err(|_| "--expected-revision must be an integer".into())
            })
            .transpose()
    }

    fn expiry(&self) -> Result<Option<i64>, String> {
        let Some(value) = self.text("expires-at")? else {
            return Ok(None);
        };
        let expiry = chrono::DateTime::parse_from_rfc3339(value)
            .map_err(|_| "--expires-at must be an RFC 3339 date, such as 2027-01-01T00:00:00Z")?
            .timestamp();
        if expiry <= chrono::Utc::now().timestamp() {
            return Err("--expires-at must be in the future".into());
        }
        Ok(Some(expiry))
    }

    fn profile(&self) -> Result<PathBuf, String> {
        let explicit = self.path("access-file");
        let inherited = std::env::var_os("OBSIDIAN_ACCESS_FILE").map(PathBuf::from);
        if let (Some(explicit), Some(inherited)) = (&explicit, &inherited)
            && explicit != inherited
        {
            return Err(
                "--access-file conflicts with OBSIDIAN_ACCESS_FILE; remove the stale setting"
                    .into(),
            );
        }
        let profile = explicit.or(inherited).ok_or_else(|| {
            "select a profile with --access-file PATH; 'access init VAULT' creates one".to_owned()
        })?;
        if !profile.is_absolute() {
            return Err("access file must be an absolute path".into());
        }
        Ok(profile)
    }

    fn store(&self) -> Result<AccessStore, String> {
        AccessStore::open(&self.profile()?).map_err(|error| error.to_string())
    }
}

async fn execute(args: &Arguments) -> Result<Output, String> {
    validate_options(args)?;
    match args.word(0)? {
        "headers" => headers(args),
        "init" => initialize(args),
        "setup" => setup(args),
        "client" => client(args),
        "grant" => add_rule(args, false),
        "deny" => add_rule(args, true),
        "rules" => remove_rule(args),
        "token" => token(args),
        "validate" => validate(args),
        "check" => check(args),
        "preview" => preview(args),
        "config" => config(args),
        "doctor" => doctor(args).await,
        "rebind" => rebind(args),
        other => Err(format!(
            "unknown access command {other}; use 'obsidian-mcp access --help'"
        )),
    }
}

fn initialize(args: &Arguments) -> Result<Output, String> {
    let vault = vault_argument(args)?;
    let profile = initialization_profile(args, &vault)?;
    let data = args
        .path("data-dir")
        .unwrap_or_else(|| default_data_dir(&profile));
    preflight_profile(&profile, &vault, &data)?;
    if args.flag("dry-run") {
        return Ok(Output::new(
            json!({"dry_run":true,"access_file":profile,"data_dir":data}),
            [format!(
                "Would initialize {}. No files were changed.",
                profile.display()
            )],
        ));
    }
    let endpoint = setup_endpoint(args)?;
    let store =
        AccessStore::initialize(&profile, &vault, &data).map_err(|error| error.to_string())?;
    let snapshot = store.snapshot().map_err(|error| error.to_string())?;
    let launch = launch_command(
        &profile,
        &snapshot.policy.source.canonical_path,
        endpoint.backend_url.as_str(),
    )?;
    Ok(Output::new(
        json!({"access_file":profile,"data_dir":data,"revision":snapshot.policy.revision}),
        [
            format!("Created access profile {}.", profile.display()),
            "Add a client with 'access client add NAME --directory PATH --role reader|editor' and the --access-file path above.".into(),
            launch,
        ],
    ))
}

fn client(args: &Arguments) -> Result<Output, String> {
    let store = args.store()?;
    match args.word(1)? {
        "list" => {
            args.require_words(2)?;
            let snapshot = store.snapshot().map_err(|error| error.to_string())?;
            let clients: Vec<Value> = snapshot
                .policy
                .clients
                .iter()
                .map(|(id, client)| client_metadata(id, client))
                .collect();
            let mut lines = Vec::new();
            for (id, client) in &snapshot.policy.clients {
                lines.push(format!(
                    "{id}: {}",
                    if client.enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                ));
                for grant in &client.grants {
                    lines.push(format!(
                        "  {}: {} {}",
                        grant.id,
                        role_name(grant.role),
                        selector(&grant.directory, &grant.file)
                    ));
                }
                for deny in &client.denies {
                    lines.push(format!(
                        "  {}: deny {}",
                        deny.id,
                        selector(&deny.directory, &deny.file)
                    ));
                }
                for credential in &client.credentials {
                    lines.push(format!(
                        "  credential {}: {}",
                        credential.id,
                        credential_status(credential)
                    ));
                }
            }
            if lines.is_empty() {
                lines.push("No clients configured. Add one with 'access client add NAME'.".into());
            }
            Ok(Output::new(
                json!({"revision":snapshot.policy.revision,"clients":clients}),
                lines,
            ))
        }
        "add" => {
            args.require_words(3)?;
            let name = args.word(2)?;
            validate_client_id(name).map_err(|error| error.to_string())?;
            let role = parse_role(args.text("role")?.unwrap_or("reader"))?;
            let expiry = args.expiry()?;
            let out = args
                .path("token-out")
                .unwrap_or_else(|| default_token_path(store.path(), name));
            store
                .validate_token_output(&out)
                .map_err(|error| error.to_string())?;
            let mut policy = ClientPolicy::default();
            append_grants(args, &mut policy, role)?;
            append_denies(args, &mut policy)?;
            let mut preview = store.snapshot().map_err(|error| error.to_string())?.policy;
            check_revision(args, preview.revision)?;
            if preview.clients.contains_key(name) {
                return Err(format!(
                    "client {name} already exists; use grant or token create"
                ));
            }
            preview.clients.insert(name.to_owned(), policy.clone());
            preview.validate().map_err(|error| error.to_string())?;
            if args.flag("dry-run") {
                return Ok(changed_output(
                    args,
                    preview,
                    format!("Would add client {name}."),
                ));
            }
            let receipt = store
                .create_client_with_token(name, &policy, &out, expiry, args.revision()?)
                .map_err(|error| format!("Client provisioning did not complete: {error}"))?;
            Ok(Output::new(
                json!({"client_id":name,"credential_id":receipt.credential_id,"token_file":receipt.path,"revision":receipt.revision}),
                [
                    format!("Created client {name}."),
                    format!(
                        "Token saved to {}. It is not printed.",
                        receipt.path.display()
                    ),
                    format!(
                        "Create client configuration: obsidian-mcp access config {name} --access-file {} --token-file {} --client codex --out CONFIG_PATH",
                        quote_path(store.path()),
                        quote_path(&receipt.path)
                    ),
                ],
            ))
        }
        "disable" | "enable" | "remove" => {
            args.require_words(3)?;
            let operation = args.word(1)?;
            let name = args.word(2)?;
            let snapshot = change_policy(&store, args, |policy| {
                if operation == "remove" {
                    policy
                        .clients
                        .remove(name)
                        .ok_or_else(|| format!("client {name} does not exist"))?;
                } else {
                    policy
                        .clients
                        .get_mut(name)
                        .ok_or_else(|| format!("client {name} does not exist"))?
                        .enabled = operation == "enable";
                }
                Ok(())
            })?;
            Ok(changed_output(
                args,
                snapshot,
                format!("Client {name}: {operation}."),
            ))
        }
        _ => Err("usage: access client add|list|enable|disable|remove [NAME]".into()),
    }
}

fn add_rule(args: &Arguments, denied: bool) -> Result<Output, String> {
    args.require_words(2)?;
    let name = args.word(1)?;
    let store = args.store()?;
    let snapshot = change_policy(&store, args, |policy| {
        let client = policy
            .clients
            .get_mut(name)
            .ok_or_else(|| format!("client {name} does not exist"))?;
        if denied {
            if args.value("directory").is_none() && args.value("file").is_none() {
                return Err("deny requires --directory PATH or --file PATH".into());
            }
            for value in option_texts(args, "directory")? {
                client.denies.push(Deny::directory(new_id("rule")?, value));
            }
            for value in option_texts(args, "file")? {
                client.denies.push(Deny::file(new_id("rule")?, value));
            }
        } else {
            let role = parse_role(args.required_text("role")?)?;
            if args.value("directory").is_none() && args.value("file").is_none() {
                return Err("grant requires --directory PATH or --file PATH".into());
            }
            append_grants(args, client, role)?;
        }
        Ok(())
    })?;
    Ok(changed_output(
        args,
        snapshot,
        format!("Updated permissions for {name}."),
    ))
}

fn remove_rule(args: &Arguments) -> Result<Output, String> {
    args.require_words(4)?;
    if args.word(1)? != "remove" {
        return Err("usage: access rules remove CLIENT RULE_ID".into());
    }
    let name = args.word(2)?;
    let rule = args.word(3)?;
    let store = args.store()?;
    let snapshot = change_policy(&store, args, |policy| {
        let client = policy
            .clients
            .get_mut(name)
            .ok_or_else(|| format!("client {name} does not exist"))?;
        let before = client.grants.len() + client.denies.len();
        client.grants.retain(|grant| grant.id != rule);
        client.denies.retain(|deny| deny.id != rule);
        if before == client.grants.len() + client.denies.len() {
            return Err(format!("rule {rule} does not exist for client {name}"));
        }
        Ok(())
    })?;
    Ok(changed_output(
        args,
        snapshot,
        format!("Removed rule {rule} from {name}."),
    ))
}

fn token(args: &Arguments) -> Result<Output, String> {
    let store = args.store()?;
    let name = args.word(2)?;
    match args.word(1)? {
        "create" => {
            args.require_words(3)?;
            let snapshot = store.snapshot().map_err(|error| error.to_string())?;
            if !snapshot.policy.clients.contains_key(name) {
                return Err(format!("client {name} does not exist"));
            }
            let out = args
                .path("out")
                .unwrap_or_else(|| default_token_path(store.path(), name));
            let expiry = args.expiry()?;
            store
                .validate_token_output(&out)
                .map_err(|error| error.to_string())?;
            if args.flag("dry-run") {
                check_revision(args, snapshot.policy.revision)?;
                return Ok(Output::new(
                    json!({"dry_run":true,"client_id":name,"token_file":out,"expires_at":expiry}),
                    [format!(
                        "Would create another credential for {name} at {}. Existing credentials remain valid.",
                        out.display()
                    )],
                ));
            }
            let receipt = store
                .issue_token_at_revision(name, &out, expiry, args.revision()?)
                .map_err(|error| error.to_string())?;
            Ok(Output::new(json!({"client_id":name,"credential_id":receipt.credential_id,"token_file":receipt.path,"revision":receipt.revision}),
                [format!("Created credential {} for {name} at {}.",receipt.credential_id,receipt.path.display()),
                 "Existing credentials remain valid. Verify the new connection, then revoke the selected old credential ID.".into()]))
        }
        "list" => {
            args.require_words(3)?;
            let snapshot = store.snapshot().map_err(|error| error.to_string())?;
            let client = snapshot
                .policy
                .clients
                .get(name)
                .ok_or_else(|| format!("client {name} does not exist"))?;
            let credentials: Vec<Value> =
                client.credentials.iter().map(credential_metadata).collect();
            let mut lines: Vec<String> = client
                .credentials
                .iter()
                .map(|credential| format!("{}: {}", credential.id, credential_status(credential)))
                .collect();
            if lines.is_empty() {
                lines.push(format!(
                    "No credentials for {name}. Create one with 'access token create {name}'."
                ));
            }
            Ok(Output::new(
                json!({"client_id":name,"credentials":credentials}),
                lines,
            ))
        }
        "revoke" => {
            args.require_words(4)?;
            let id = args.word(3)?;
            let change = |policy: &mut AccessPolicy| {
                let credential = policy
                    .clients
                    .get_mut(name)
                    .ok_or_else(|| format!("client {name} does not exist"))?
                    .credentials
                    .iter_mut()
                    .find(|credential| credential.id == id)
                    .ok_or_else(|| format!("credential {id} does not exist for {name}"))?;
                credential.revoked = true;
                Ok(())
            };
            let snapshot = change_policy(&store, args, change)?;
            Ok(changed_output(
                args,
                snapshot,
                format!("Revoked credential {id} for {name}."),
            ))
        }
        _ => Err(
            "usage: access token create|list CLIENT, or access token revoke CLIENT CREDENTIAL_ID"
                .into(),
        ),
    }
}

fn validate(args: &Arguments) -> Result<Output, String> {
    args.require_words(1)?;
    let store = args.store()?;
    let snapshot = store.snapshot().map_err(|error| error.to_string())?;
    snapshot
        .policy
        .source
        .verify()
        .map_err(|error| error.to_string())?;
    Ok(Output::new(
        json!({"valid":true,"revision":snapshot.policy.revision,"clients":snapshot.policy.clients.len()}),
        [format!(
            "Access profile is valid (revision {}).",
            snapshot.policy.revision
        )],
    ))
}

fn check(args: &Arguments) -> Result<Output, String> {
    args.require_words(2)?;
    let name = args.word(1)?;
    let action = parse_action(args.required_text("action")?)?;
    let path = args.required_text("path")?;
    let snapshot = args
        .store()?
        .snapshot()
        .map_err(|error| error.to_string())?;
    let client = snapshot
        .policy
        .clients
        .get(name)
        .ok_or_else(|| format!("client {name} does not exist"))?;
    snapshot
        .policy
        .source
        .verify()
        .map_err(|error| error.to_string())?;
    let mut decision = client.decision(action, Path::new(path));
    let mut resolved_path = None;
    if decision.allowed {
        let root = VaultRoot::open(&snapshot.policy.source.canonical_path)
            .map_err(|error| error.to_string())?;
        let actual = root
            .resolve_for_write(Path::new(path))
            .map_err(|error| error.to_string())?;
        let physical_decision = client.decision(action, &actual.relative);
        if !physical_decision.allowed {
            decision = physical_decision;
        }
        resolved_path = Some(actual.relative);
    }
    let reason = match decision.reason {
        DecisionReason::Allowed => "allowed by grant",
        DecisionReason::Disabled => "client disabled",
        DecisionReason::ReservedPath => "owner metadata is reserved",
        DecisionReason::Denied => "denied by path rule (case/Unicode protection applies)",
        DecisionReason::NoGrant => "no matching grant",
        DecisionReason::InvalidPath => "invalid resource path",
    };
    let rule = decision
        .rule_id
        .as_deref()
        .map(|id| format!(" ({id})"))
        .unwrap_or_default();
    let line = format!(
        "{}: {reason}{rule}",
        if decision.allowed {
            "Allowed"
        } else {
            "Denied"
        }
    );
    Ok(Output::new(
        json!({"client_id":name,"action":args.required_text("action")?,"path":path,"resolved_path":resolved_path,"decision":decision,"revision":snapshot.policy.revision}),
        [line],
    ))
}

fn preview(args: &Arguments) -> Result<Output, String> {
    args.require_words(2)?;
    let name = args.word(1)?;
    let snapshot = args
        .store()?
        .snapshot()
        .map_err(|error| error.to_string())?;
    snapshot
        .policy
        .source
        .verify()
        .map_err(|error| error.to_string())?;
    let client = snapshot
        .policy
        .clients
        .get(name)
        .ok_or_else(|| format!("client {name} does not exist"))?;
    let limit = args
        .text("max-results")?
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| "--max-results must be a positive integer")?
        .unwrap_or(100);
    if !(1..=MAX_PREVIEW).contains(&limit) {
        return Err(format!("--max-results must be from 1 to {MAX_PREVIEW}"));
    }
    let root = VaultRoot::open(&snapshot.policy.source.canonical_path)
        .map_err(|error| error.to_string())?;
    let mut paths = Vec::new();
    for path in root
        .list_files(Path::new(""), true, None)
        .map_err(|error| error.to_string())?
    {
        if !client.allows(AccessAction::Read, &path) {
            continue;
        }
        let eligible = || {
            let actual = root.resolve_existing(&path)?;
            if !client.allows(AccessAction::Read, &actual.relative)
                || !root.entry_metadata_exact(&actual.relative)?.is_file()
            {
                return Ok(false);
            }
            root.validate_access_file(&actual.relative)?;
            Ok(true)
        };
        match eligible() {
            Ok(true) => paths.push(path),
            Ok(false) => (),
            Err(error) if unavailable_preview_entry(&error) => (),
            Err(error) => return Err(error.to_string()),
        }
    }
    let total = paths.len();
    paths.truncate(limit);
    let mut lines = vec![format!("{name}: {total} permitted file path(s).")];
    lines.extend(paths.iter().map(|path| path.display().to_string()));
    if total > limit {
        lines.push(format!(
            "Showing {limit}; use --max-results up to {MAX_PREVIEW}."
        ));
    }
    Ok(Output::new(
        json!({"client_id":name,"paths":paths,"total":total,"truncated":total>limit,"revision":snapshot.policy.revision}),
        lines,
    ))
}

fn unavailable_preview_entry(error: &VaultError) -> bool {
    matches!(
        error,
        VaultError::NoteNotFound(_)
            | VaultError::DirectoryNotFound(_)
            | VaultError::OutsideVault(_)
            | VaultError::InvalidPath(_)
            | VaultError::AccessDenied { .. }
    ) || matches!(error, VaultError::Io(error) if matches!(error.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory))
}

fn rebind(args: &Arguments) -> Result<Output, String> {
    args.require_words(1)?;
    let vault = args.path("vault").ok_or("rebind requires --vault PATH")?;
    let source = SourceIdentity::capture(&vault).map_err(|error| error.to_string())?;
    let store = args.store()?;
    let snapshot = change_policy(&store, args, |policy| {
        policy.source = source;
        Ok(())
    })?;
    let mut output = changed_output(args, snapshot, "Updated vault binding.".into());
    output.lines.push("Restart the authenticated server to open the new binding. Existing servers cannot inherit it.".into());
    Ok(output)
}

fn headers(args: &Arguments) -> Result<Output, String> {
    args.require_words(1)?;
    let file = args
        .path("token-file")
        .ok_or("headers requires --token-file ABSOLUTE_PATH")?;
    let token = read_token_file(&file).map_err(|error| error.to_string())?;
    Ok(Output {
        value: json!({"Authorization":format!("Bearer {}",token.expose())}),
        lines: Vec::new(),
        protocol: true,
    })
}

fn config(args: &Arguments) -> Result<Output, String> {
    args.require_words(2)?;
    let name = args.word(1)?;
    let store = args.store()?;
    let token_file = args.path("token-file").ok_or(
        "config requires --token-file PATH; use the file returned by client add or token create",
    )?;
    let token = read_token_file(&token_file).map_err(|error| error.to_string())?;
    let authenticated = store
        .authenticate(token.expose())
        .map_err(|error| error.to_string())?;
    if authenticated.client_id() != name {
        return Err("token file belongs to another client".into());
    }
    let adapter = args.text("client")?.unwrap_or("codex");
    let transport = parse_transport(args.text("transport")?.unwrap_or("http"))?;
    let client_token = args.path("client-token-file").unwrap_or(token_file);
    if !absolute_client_path(&client_token) {
        return Err("--client-token-file must be absolute on the client machine".into());
    }
    let binary = args
        .path("binary")
        .unwrap_or(std::env::current_exe().map_err(|error| error.to_string())?);
    let out = args
        .path("out")
        .ok_or("config requires --out PATH; existing configuration is never overwritten")?;
    store
        .validate_token_output(&out)
        .map_err(|error| error.to_string())?;
    let bytes = if transport == "stdio" {
        if args.value("url").is_some() || args.value("client-token-file").is_some() {
            return Err("restricted stdio uses the local vault, policy and token file; omit --url and --client-token-file".into());
        }
        stdio_configuration_bytes(
            name,
            adapter,
            &binary,
            store.path(),
            &client_token,
            &authenticated.source().canonical_path,
        )?
    } else {
        let url = validated_url(args.text("url")?.unwrap_or(DEFAULT_URL))?;
        configuration_bytes(name, adapter, url.as_str(), &binary, &client_token)?
    };
    if !args.flag("dry-run") {
        write_protected_file(&out, &bytes).map_err(|error| error.to_string())?;
    }
    let mut lines = vec![
        format!("{} {adapter} {transport} configuration at {}.", if args.flag("dry-run") { "Would create" } else { "Created" }, out.display()),
        "Merge this snippet into the client's MCP configuration, then reconnect. No shell environment variable is needed.".into(),
    ];
    if transport == "http" {
        if args.value("client-token-file").is_some() {
            lines.push(format!(
                "On the client machine, place the token at {} using your secure transfer method.",
                client_token.display()
            ));
        }
        lines.push("This uses the client's native header helper. Verify support in the installed client; other applications are not implied to be compatible.".into());
    } else {
        lines.push("The client starts a restricted server process on this machine. Its OS account still controls any filesystem access it already has.".into());
    }
    Ok(Output::new(
        json!({"client_id":name,"adapter":adapter,"transport":transport,"configuration_file":out,"client_token_file":client_token,"dry_run":args.flag("dry-run")}),
        lines,
    ))
}

fn configuration_bytes(
    name: &str,
    adapter: &str,
    url: &str,
    binary: &Path,
    token: &Path,
) -> Result<Vec<u8>, String> {
    let helper = helper_command(binary, token)?;
    match adapter {
        "codex" => {
            let table = serde_json::to_string(name).map_err(|error| error.to_string())?;
            let url = serde_json::to_string(url).map_err(|error| error.to_string())?;
            let helper = serde_json::to_string(&helper).map_err(|error| error.to_string())?;
            Ok(
                format!("[mcp_servers.{table}]\nurl = {url}\nhttp_headers_helper = {helper}\n")
                    .into_bytes(),
            )
        }
        "claude-code" => serde_json::to_vec_pretty(
            &json!({"mcpServers":{name:{"type":"http","url":url,"headersHelper":helper}}}),
        )
        .map_err(|error| error.to_string()),
        _ => Err(format!(
            "unsupported client {adapter}; supported configuration formats: codex, claude-code. OAuth-only clients need a separate integration."
        )),
    }
}

fn stdio_configuration_bytes(
    name: &str,
    adapter: &str,
    binary: &Path,
    profile: &Path,
    token: &Path,
    vault: &Path,
) -> Result<Vec<u8>, String> {
    if !binary.is_absolute() {
        return Err("stdio binary must be an absolute local path".into());
    }
    let arguments = [
        "--access-file".to_owned(),
        profile.display().to_string(),
        "--access-token-file".to_owned(),
        token.display().to_string(),
        vault.display().to_string(),
    ];
    match adapter {
        "codex" => Ok(format!(
            "[mcp_servers.{}]\ncommand = {}\nargs = {}\nenv = {{ OBSIDIAN_TRANSPORT = \"stdio\" }}\n",
            serde_json::to_string(name).map_err(|error| error.to_string())?,
            serde_json::to_string(&binary).map_err(|error| error.to_string())?,
            serde_json::to_string(&arguments).map_err(|error| error.to_string())?,
        )
        .into_bytes()),
        "claude-code" => serde_json::to_vec_pretty(
            &json!({"mcpServers": {name: {"type":"stdio","command":binary,"args":arguments,"env":{"OBSIDIAN_TRANSPORT":"stdio"}}}}),
        )
        .map_err(|error| error.to_string()),
        _ => Err("supported configuration formats: codex, claude-code".into()),
    }
}

async fn doctor(args: &Arguments) -> Result<Output, String> {
    args.require_words(1)?;
    let store = args.store()?;
    let snapshot = store.snapshot().map_err(|error| error.to_string())?;
    snapshot
        .policy
        .source
        .verify()
        .map_err(|error| error.to_string())?;
    let token_file = args
        .path("token-file")
        .ok_or("doctor requires --token-file PATH")?;
    let token = read_token_file(&token_file).map_err(|error| error.to_string())?;
    store
        .authenticate(token.expose())
        .map_err(|error| error.to_string())?;
    let url = validated_url(args.text("url")?.unwrap_or(DEFAULT_URL))?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|error| error.to_string())?;
    let unreachable = "Could not reach the MCP endpoint. Start the server with the generated --access-file command and check --url.";
    let anonymous = rpc_request(&client, &url, None, "server/discover", json!({}))
        .send()
        .await
        .map_err(|_| unreachable.to_owned())?;
    if anonymous.status() != reqwest::StatusCode::UNAUTHORIZED {
        return Err(format!(
            "The endpoint returned HTTP {} without a token; it has not proved that authentication is required. Check --url and the server's --access-file setting.",
            anonymous.status().as_u16()
        ));
    }
    let discovery = rpc(&client, &url, token.expose(), "server/discover", json!({})).await?;
    if discovery.get("result").is_none() {
        return Err("Authenticated MCP discovery did not return a result; inspect the server's protected diagnostics.".into());
    }
    let catalog = rpc(&client, &url, token.expose(), "tools/list", json!({})).await?;
    let tools = catalog["result"]["tools"]
        .as_array()
        .ok_or("Authenticated tool discovery did not return a catalog")?;
    let mut checks = vec![
        "profile valid".to_owned(),
        "unauthenticated MCP refused".into(),
        "authenticated MCP discovery and tool catalog passed".into(),
    ];
    if let Some(path) = args.text("path")? {
        rpc(
            &client,
            &url,
            token.expose(),
            "tools/call",
            json!({"name":"note_read","arguments":{"path":path,"max_bytes":1024}}),
        )
        .await?;
        checks.push("requested note read passed; content was not printed".into());
    }
    if args.flag("semantic") {
        rpc(&client,&url,token.expose(),"tools/call",json!({"name":"search_semantic","arguments":{"query":"connection setup check","top_k":1}})).await?;
        checks.push("explicit semantic query passed".into());
    }
    if let Some(path) = args.text("write-check")? {
        if !path.ends_with(".md") {
            return Err("--write-check requires a disposable .md path".into());
        }
        let marker = format!("Obsidian MCP disposable access check {}", new_id("probe")?);
        rpc(
            &client,
            &url,
            token.expose(),
            "tools/call",
            json!({"name":"note_create","arguments":{"path":path,"content":marker}}),
        )
        .await?;
        let read = rpc(
            &client,
            &url,
            token.expose(),
            "tools/call",
            json!({"name":"note_read","arguments":{"path":path}}),
        )
        .await;
        let exact = read.as_ref().is_ok_and(|body| {
            body["result"]["content"].as_array().is_some_and(|items| {
                items
                    .iter()
                    .any(|item| item["text"].as_str() == Some(marker.as_str()))
            })
        });
        if !exact {
            return Err(format!(
                "Created disposable note {path}, but exact readback failed. Inspect it before removing it; the check did not claim success."
            ));
        }
        rpc(
            &client,
            &url,
            token.expose(),
            "tools/call",
            json!({"name":"note_delete","arguments":{"path":path,"confirm":true}}),
        )
        .await
        .map_err(|error| {
            format!("Created and read disposable note {path}, but cleanup failed: {error}")
        })?;
        checks.push("explicit disposable create/read/delete passed".into());
    }
    let mut lines = vec!["MCP connection check passed.".into()];
    lines.extend(checks.iter().map(|check| format!("- {check}")));
    lines.push("This checks the endpoint directly. Reconnect the actual MCP client to verify its saved configuration.".into());
    Ok(Output::new(
        json!({"ok":true,"checks":checks,"tool_count":tools.len(),"real_client_verified":false}),
        lines,
    ))
}

fn rpc_request(
    client: &reqwest::Client,
    url: &reqwest::Url,
    token: Option<&str>,
    method: &str,
    mut params: Value,
) -> reqwest::RequestBuilder {
    params["_meta"] = json!({"io.modelcontextprotocol/protocolVersion":MODERN,
        "io.modelcontextprotocol/clientInfo":{"name":"obsidian-mcp-access-check","version":env!("CARGO_PKG_VERSION")},
        "io.modelcontextprotocol/clientCapabilities":{}});
    let mut request = client
        .post(url.clone())
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .header("MCP-Protocol-Version", MODERN)
        .header("Mcp-Method", method);
    if method == "tools/call"
        && let Some(name) = params["name"].as_str()
    {
        request = request.header("Mcp-Name", name);
    }
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    request.body(
        json!({"jsonrpc":"2.0","id":"access-doctor","method":method,"params":params}).to_string(),
    )
}

async fn rpc(
    client: &reqwest::Client,
    url: &reqwest::Url,
    token: &str,
    method: &str,
    params: Value,
) -> Result<Value, String> {
    let mut response = rpc_request(client, url, Some(token), method, params)
        .send()
        .await
        .map_err(|_| "MCP request failed; check the endpoint and server diagnostics".to_owned())?;
    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Err("The server rejected this token. Check the selected profile, token expiry, client status and credential revocation.".into());
    }
    if !response.status().is_success() {
        return Err(format!(
            "MCP endpoint returned HTTP {}; inspect protected server diagnostics",
            response.status().as_u16()
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "MCP response ended before completion")?
    {
        if bytes.len() + chunk.len() > MAX_HTTP_BYTES {
            return Err("MCP diagnostic response exceeded its size limit".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    let body = decode_rpc_response(&bytes)?;
    if let Some(error) = body.get("error") {
        match error["data"]["code"].as_str() {
            Some("semantic_warming") => return Err("Authentication passed; semantic search is warming. Retry the explicit --semantic check shortly.".into()),
            Some("permission_denied") => return Err("Authentication passed, but the client cannot perform this action. Check its role and grants with 'access check'.".into()),
            Some("authorization_unavailable") => return Err("The server cannot validate its access policy. Run 'access validate' on the server and repair the reported store or vault binding.".into()),
            _ => (),
        }
        let code = error
            .get("code")
            .and_then(Value::as_i64)
            .map(|code| code.to_string())
            .unwrap_or_else(|| "unknown".into());
        return Err(format!(
            "MCP operation failed (code {code}); check effective permissions and protected server diagnostics"
        ));
    }
    if body["result"]["isError"] == true {
        return Err("MCP tool reported a failure; check effective permissions and protected server diagnostics".into());
    }
    Ok(body)
}

fn setup(args: &Arguments) -> Result<Output, String> {
    let vault = vault_argument(args)?;
    let source = SourceIdentity::capture(&vault).map_err(|error| error.to_string())?;
    let profile = initialization_profile(args, &vault)?;
    let data = args
        .path("data-dir")
        .unwrap_or_else(|| default_data_dir(&profile));
    preflight_profile(&profile, &vault, &data)?;
    let interactive = io::stdin().is_terminal();
    let missing_name = "Noninteractive setup needs --name NAME; select grants with --directory PATH or --file PATH, or run access setup in a terminal.";
    let name = match args.text("name")? {
        Some(name) => name.to_owned(),
        None if interactive => prompt("Client name (letters, digits, - or _)", Some("assistant"))?,
        None => return Err(missing_name.into()),
    };
    validate_client_id(&name).map_err(|error| error.to_string())?;
    let role = match args.text("role")? {
        Some(role) => parse_role(role)?,
        None if interactive => {
            parse_role(&prompt_choice("Role", "reader", &["reader", "editor"])?)?
        }
        None => Role::Reader,
    };
    let expiry = args.expiry()?;
    let mut policy = ClientPolicy::default();
    append_grants(args, &mut policy, role)?;
    append_denies(args, &mut policy)?;
    if policy.grants.is_empty() && interactive {
        let selection = prompt(
            "Shared directory (use / for the whole vault; leave blank for no access)",
            Some(""),
        )?;
        if !selection.is_empty() {
            policy.grants.push(Grant::directory(
                new_id("rule")?,
                if selection == "/" { "" } else { &selection },
                role,
            ));
        }
        let denied = prompt("Denied directory (optional)", Some(""))?;
        if !denied.is_empty() {
            policy.denies.push(Deny::directory(new_id("rule")?, denied));
        }
    }
    let adapter = match args.text("client")? {
        Some(client) => client.to_owned(),
        None if interactive => {
            prompt_choice("Client configuration", "codex", &["codex", "claude-code"])?
        }
        None => "codex".into(),
    };
    if !matches!(adapter.as_str(), "codex" | "claude-code") {
        return Err("Supported setup clients: codex, claude-code. OAuth-only applications need a separate integration.".into());
    }
    let transport = match args.text("transport")? {
        Some(value) => parse_transport(value)?,
        None if interactive => {
            parse_transport(&prompt_choice("Connection", "http", &["http", "stdio"])?)?
        }
        None => "http",
    };
    if transport == "stdio"
        && (args.value("url").is_some() || args.value("client-token-file").is_some())
    {
        return Err(
            "restricted stdio setup runs on this machine; omit --url and --client-token-file"
                .into(),
        );
    }
    let endpoint = if transport == "http" {
        Some(setup_endpoint(args)?)
    } else {
        None
    };
    let binary = args
        .path("binary")
        .unwrap_or(std::env::current_exe().map_err(|error| error.to_string())?);
    let token_path = args
        .path("token-out")
        .unwrap_or_else(|| default_token_path(&profile, &name));
    let client_token = args
        .path("client-token-file")
        .unwrap_or_else(|| token_path.clone());
    if !absolute_client_path(&client_token) {
        return Err("--client-token-file must be absolute on the client machine".into());
    }
    let config_path = args.path("out").unwrap_or_else(|| {
        profile.parent().unwrap_or(Path::new("/")).join(format!(
            "{name}-{}.{}",
            adapter,
            if adapter == "codex" { "toml" } else { "json" }
        ))
    });
    if profile == token_path || profile == config_path || token_path == config_path {
        return Err("profile, token and client configuration must use different files".into());
    }
    validate_initialization(&token_path, &vault, &data).map_err(|error| error.to_string())?;
    validate_initialization(&config_path, &vault, &data).map_err(|error| error.to_string())?;
    let mut planned = AccessPolicy::new(source.clone(), data.clone());
    planned.clients.insert(name.clone(), policy.clone());
    planned.validate().map_err(|error| error.to_string())?;
    if interactive {
        eprintln!(
            "Client {name}; role {}; grants {}; denials {}.",
            role_name(role),
            policy.grants.len(),
            policy.denies.len()
        );
        for grant in &policy.grants {
            eprintln!(
                "  {} {}",
                role_name(grant.role),
                selector(&grant.directory, &grant.file)
            );
        }
        for deny in &policy.denies {
            eprintln!("  deny {}", selector(&deny.directory, &deny.file));
        }
        eprintln!(
            "Profile: {}\nToken delivery file: {}\nClient-local token file: {}",
            profile.display(),
            token_path.display(),
            client_token.display()
        );
        if !args.flag("dry-run")
            && prompt_choice("Create this setup", "no", &["yes", "no"])? != "yes"
        {
            return Err("Setup cancelled; no files were created.".into());
        }
    }
    let configuration = if let Some(endpoint) = &endpoint {
        configuration_bytes(
            &name,
            &adapter,
            endpoint.public_url.as_str(),
            &binary,
            &client_token,
        )?
    } else {
        stdio_configuration_bytes(
            &name,
            &adapter,
            &binary,
            &profile,
            &client_token,
            &source.canonical_path,
        )?
    };
    let launch = endpoint
        .as_ref()
        .map(|endpoint| {
            launch_command(
                &profile,
                &source.canonical_path,
                endpoint.backend_url.as_str(),
            )
        })
        .transpose()?;
    let backend_doctor = endpoint
        .as_ref()
        .map(|endpoint| doctor_command(&profile, &token_path, &endpoint.backend_url))
        .transpose()?;
    let public_doctor = endpoint
        .as_ref()
        .filter(|endpoint| endpoint.public_url.scheme() == "https")
        .map(|endpoint| doctor_command(&profile, &token_path, &endpoint.public_url))
        .transpose()?;
    let proxy_endpoint = endpoint
        .as_ref()
        .filter(|endpoint| endpoint.public_url.scheme() == "https");
    let proxy_host_header = proxy_endpoint
        .map(SetupEndpoint::backend_host_header)
        .transpose()?;
    let proxy_route = proxy_endpoint
        .zip(proxy_host_header.as_ref())
        .map(|(endpoint, host)| {
            format!(
                "Route your HTTPS proxy's {} endpoint to {} and forward Host as {host}.",
                endpoint.public_url, endpoint.backend_url
            )
        });
    let mut result = json!({"client_id":name,"access_file":profile,"token_file":token_path,
        "client_token_file":client_token,"configuration_file":config_path,"transport":transport,
        "url":endpoint.as_ref().map(|endpoint|endpoint.public_url.as_str()),
        "backend_url":endpoint.as_ref().map(|endpoint|endpoint.backend_url.as_str()),
        "server_command":launch,"doctor_command":backend_doctor,"public_doctor_command":public_doctor,
        "proxy_upstream":proxy_endpoint.map(|endpoint|endpoint.backend_url.as_str()),
        "proxy_host_header":proxy_host_header});
    if args.flag("dry-run") {
        let mut lines = vec!["Setup preview complete; no files were created.".into()];
        if let Some(launch) = &launch {
            lines.push(launch.clone());
            if let Some(route) = &proxy_route {
                lines.push(route.clone());
            }
            if let Some(doctor) = &backend_doctor {
                lines.push(format!("Check the backend: {doctor}"));
            }
            if let Some(doctor) = &public_doctor {
                lines.push(format!(
                    "After configuring the proxy, check its public endpoint: {doctor}"
                ));
            }
        } else {
            lines.push(format!(
                "Would create {} for {adapter}. The client starts the restricted server process.",
                config_path.display()
            ));
        }
        result["dry_run"] = json!(true);
        return Ok(Output::new(result, lines));
    }
    let store =
        AccessStore::initialize(&profile, &vault, &data).map_err(|error| error.to_string())?;
    let delivery_error = |error| {
        format!(
            "Profile was created but client provisioning did not complete: {error}. Existing servers were not changed."
        )
    };
    let receipt = store
        .create_client_with_token(&name, &policy, &token_path, expiry, Some(planned.revision))
        .map_err(delivery_error)?;
    let configuration_error = |error| {
        format!(
            "Credential {} was delivered at {}, but client configuration was not created: {error}. Use 'access config' with that token file.",
            receipt.credential_id,
            receipt.path.display()
        )
    };
    write_protected_file(&config_path, &configuration).map_err(configuration_error)?;
    let mut lines = vec![format!(
        "Created client {name} and saved its token to {}.",
        receipt.path.display()
    )];
    if let Some(launch) = &launch {
        lines.push(format!("1. Start the authenticated server: {launch}"));
        if let Some(route) = &proxy_route {
            lines.push(format!("   {route}"));
        }
        lines.push(format!(
            "2. Merge {} into the {adapter} MCP configuration and reconnect.",
            config_path.display()
        ));
        if args.value("client-token-file").is_some() {
            lines.push(format!("   On the client machine, place the token at {} using your secure transfer method.", client_token.display()));
        }
        if let Some(doctor) = &backend_doctor {
            lines.push(format!("3. Check the authenticated backend: {doctor}"));
        }
        if let Some(doctor) = &public_doctor {
            lines.push(format!(
                "   After configuring the proxy, check its public endpoint: {doctor}"
            ));
        }
    } else {
        lines.push(format!("Merge {} into the {adapter} MCP configuration and reconnect. The client starts the restricted server process.", config_path.display()));
        lines.push("This delegates MCP permissions; the launching OS account retains its existing filesystem authority.".into());
    }
    lines.push("Existing unauthenticated servers and client settings were not changed. Embedding settings were not enabled or changed.".into());

    result["credential_id"] = json!(receipt.credential_id);
    result["token_file"] = json!(receipt.path);
    result["revision"] = json!(receipt.revision);
    Ok(Output::new(result, lines))
}

fn validate_options(args: &Arguments) -> Result<(), String> {
    let command = args.word(0)?;
    let mut allowed = vec!["access-file", "json", "help"];
    let mutation = ["dry-run", "expected-revision"];
    match command {
        "init" => allowed.extend(["vault", "data-dir", "dry-run"]),
        "setup" => allowed.extend([
            "vault",
            "data-dir",
            "directory",
            "file",
            "deny-directory",
            "deny-file",
            "role",
            "name",
            "client",
            "url",
            "token-out",
            "out",
            "expires-at",
            "client-token-file",
            "binary",
            "transport",
            "dry-run",
        ]),
        "client" => match args.word(1)? {
            "add" => {
                allowed.extend(mutation);
                allowed.extend([
                    "directory",
                    "file",
                    "deny-directory",
                    "deny-file",
                    "role",
                    "token-out",
                    "expires-at",
                ]);
            }
            "enable" | "disable" | "remove" => allowed.extend(mutation),
            "list" => (),
            _ => return Err("usage: access client add|list|enable|disable|remove [NAME]".into()),
        },
        "grant" => {
            allowed.extend(mutation);
            allowed.extend(["directory", "file", "role"]);
        }
        "deny" => {
            allowed.extend(mutation);
            allowed.extend(["directory", "file"]);
        }
        "rules" | "rebind" => {
            allowed.extend(mutation);
            if command == "rebind" {
                allowed.push("vault");
            }
        }
        "token" => match args.word(1)? {
            "create" => {
                allowed.extend(mutation);
                allowed.extend(["out", "expires-at"]);
            }
            "revoke" => allowed.extend(mutation),
            "list" => (),
            _ => {
                return Err(
                    "usage: access token create|list CLIENT, or token revoke CLIENT ID".into(),
                );
            }
        },
        "validate" => (),
        "check" => allowed.extend(["action", "path"]),
        "preview" => allowed.push("max-results"),
        "config" => allowed.extend([
            "client",
            "url",
            "token-file",
            "client-token-file",
            "binary",
            "out",
            "transport",
            "dry-run",
        ]),
        "headers" => allowed.push("token-file"),
        "doctor" => allowed.extend(["url", "token-file", "path", "semantic", "write-check"]),
        _ => return Ok(()),
    }
    for option in args.values.keys().chain(args.flags.iter()) {
        if !allowed.contains(&option.as_str()) {
            return Err(format!("--{option} is not valid for access {command}"));
        }
    }
    Ok(())
}

fn decode_rpc_response(bytes: &[u8]) -> Result<Value, String> {
    if let Ok(value) = serde_json::from_slice::<Value>(bytes) {
        return if value["id"] == "access-doctor" {
            Ok(value)
        } else {
            Err("MCP endpoint returned a response for another request".into())
        };
    }
    let text =
        std::str::from_utf8(bytes).map_err(|_| "MCP endpoint returned an invalid response")?;
    for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
        let value: Value = serde_json::from_str(data)
            .map_err(|_| "MCP endpoint returned an invalid event response")?;
        if value["id"] == "access-doctor" {
            return Ok(value);
        }
    }
    Err("MCP endpoint returned no response for the connection check".into())
}

fn windows_path(path: &Path) -> bool {
    let text = path.to_string_lossy();
    let bytes = text.as_bytes();
    text.starts_with("\\\\")
        || (bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && matches!(bytes[2], b'\\' | b'/'))
}

fn absolute_client_path(path: &Path) -> bool {
    windows_path(path) || path.to_string_lossy().starts_with('/')
}

fn helper_command(binary: &Path, token: &Path) -> Result<String, String> {
    if !absolute_client_path(binary) || !absolute_client_path(token) {
        return Err("client binary and token paths must be absolute on the client machine".into());
    }
    if windows_path(binary) != windows_path(token) {
        return Err("client binary and token paths must use the same operating system's path format; set --binary and --client-token-file for the client machine".into());
    }
    if windows_path(binary) {
        let quote = |path: &Path| -> Result<String, String> {
            let value = path
                .to_str()
                .ok_or("client helper paths must be valid UTF-8")?;
            if value.chars().any(char::is_control)
                || value.contains(['"', '%', '!', '^', '&', '|', '<', '>', '$', '`'])
            {
                return Err("Windows helper paths contain shell metacharacters; choose a protected token path without those characters".into());
            }
            Ok(format!("'{}'", value.replace('\'', "''")))
        };
        Ok(format!(
            "powershell.exe -NoLogo -NoProfile -NonInteractive -Command \"& {} access headers --token-file {}\"",
            quote(binary)?,
            quote(token)?
        ))
    } else {
        let binary = binary
            .to_str()
            .ok_or("client helper paths must be valid UTF-8")?;
        let token = token
            .to_str()
            .ok_or("client helper paths must be valid UTF-8")?;
        if binary.chars().chain(token.chars()).any(char::is_control) {
            return Err("client helper paths must not contain control characters".into());
        }
        Ok(format!(
            "{} access headers --token-file {}",
            quote_posix(binary),
            quote_posix(token)
        ))
    }
}

struct SetupEndpoint {
    public_url: reqwest::Url,
    backend_url: reqwest::Url,
}

impl SetupEndpoint {
    fn backend_host_header(&self) -> Result<String, String> {
        let host = self
            .backend_url
            .host_str()
            .ok_or("backend endpoint has no host")?;
        let port = self
            .backend_url
            .port_or_known_default()
            .ok_or("backend endpoint has no port")?;
        Ok(format!("{host}:{port}"))
    }
}

fn setup_endpoint(args: &Arguments) -> Result<SetupEndpoint, String> {
    let public_url = validated_url(args.text("url")?.unwrap_or(DEFAULT_URL))?;
    let direct = public_url.scheme() == "http";
    if direct && public_url.path() != "/mcp" {
        return Err("Direct HTTP setup uses the server's /mcp path. Use /mcp here; 'access config --url' can select a separately configured proxy path.".into());
    }
    let host = if direct {
        local_setup_host(&public_url)
    } else {
        "127.0.0.1"
    };
    let preferred = if direct {
        public_url
            .port_or_known_default()
            .ok_or("setup endpoint has no port")?
    } else {
        37842
    };
    let port = if available_port(host, preferred)? {
        preferred
    } else {
        if direct && args.value("url").is_some() {
            return Err(format!(
                "Port {preferred} is already in use. Existing servers were not changed; choose another loopback port with --url http://127.0.0.1:PORT/mcp."
            ));
        }
        let mut selected = None;
        for candidate in 37843..=37942 {
            if available_port(host, candidate)? {
                selected = Some(candidate);
                break;
            }
        }
        selected.ok_or("No free setup backend port was found; free a loopback port in 37842–37942, or set up a local HTTP endpoint first.")?
    };
    let authority = if host == "::1" {
        format!("[::1]:{port}")
    } else {
        format!("{host}:{port}")
    };
    let backend_url = reqwest::Url::parse(&format!("http://{authority}/mcp"))
        .map_err(|_| "could not construct the setup backend endpoint")?;
    Ok(SetupEndpoint {
        public_url: if direct {
            backend_url.clone()
        } else {
            public_url
        },
        backend_url,
    })
}

fn available_port(host: &str, port: u16) -> Result<bool, String> {
    let endpoint = std::net::SocketAddr::new(
        host.parse()
            .map_err(|error| format!("Cannot use local host {host}: {error}"))?,
        port,
    );
    match crate::http_process::bind_available_tcp(endpoint) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::AddrInUse => Ok(false),
        Err(error) => Err(format!("Cannot use local endpoint {host}:{port}: {error}")),
    }
}

fn local_setup_host(url: &reqwest::Url) -> &'static str {
    if url.scheme() == "http" && url.host_str() == Some("[::1]") {
        "::1"
    } else {
        "127.0.0.1"
    }
}

fn vault_argument(args: &Arguments) -> Result<PathBuf, String> {
    if let Some(vault) = args.path("vault") {
        args.require_words(1)?;
        Ok(vault)
    } else {
        args.require_words(2)?;
        Ok(PathBuf::from(&args.words[1]))
    }
}

fn preflight_profile(profile: &Path, vault: &Path, data: &Path) -> Result<(), String> {
    validate_initialization(profile, vault, data).map_err(|error| {
        if matches!(&error, AccessError::Io(error) if error.kind() == io::ErrorKind::AlreadyExists) {
            format!("The profile path {} already exists and was not overwritten. To add another client, use 'obsidian-mcp access client add NAME --access-file {} --directory SHARED_FOLDER', then 'access config'. If this is not an access profile, choose another --access-file path.", profile.display(), quote_path(profile))
        } else {
            error.to_string()
        }
    })
}

fn initialization_profile(args: &Arguments, vault: &Path) -> Result<PathBuf, String> {
    if args.value("access-file").is_some() || std::env::var_os("OBSIDIAN_ACCESS_FILE").is_some() {
        args.profile()
    } else {
        default_profile_path(vault).map_err(|error| error.to_string())
    }
}

fn change_policy(
    store: &AccessStore,
    args: &Arguments,
    change: impl FnOnce(&mut AccessPolicy) -> Result<(), String>,
) -> Result<AccessPolicy, String> {
    let snapshot = store.snapshot().map_err(|error| error.to_string())?;
    check_revision(args, snapshot.policy.revision)?;
    let expected = snapshot.policy.revision;
    let mut prepared = snapshot.policy;
    change(&mut prepared)?;
    prepared.validate().map_err(|error| error.to_string())?;
    if args.flag("dry-run") {
        return Ok(prepared);
    }
    store
        .update(Some(expected), |current| {
            *current = prepared;
            Ok(())
        })
        .map(|snapshot| snapshot.policy)
        .map_err(|error| error.to_string())
}

fn check_revision(args: &Arguments, current: u64) -> Result<(), String> {
    if let Some(expected) = args.revision()?
        && expected != current
    {
        return Err(format!(
            "policy revision conflict: expected {expected}, current {current}; review the current policy and retry"
        ));
    }
    Ok(())
}

fn changed_output(args: &Arguments, policy: AccessPolicy, message: String) -> Output {
    let mut lines = vec![message];
    if args.flag("dry-run") {
        lines.push("Dry run complete; no files were changed.".into());
    } else {
        lines.push(format!(
            "Policy revision {}. Reconnect affected legacy MCP clients.",
            policy.revision
        ));
    }
    Output::new(
        json!({"dry_run":args.flag("dry-run"),"revision":policy.revision}),
        lines,
    )
}

fn append_grants(args: &Arguments, client: &mut ClientPolicy, role: Role) -> Result<(), String> {
    for value in option_texts(args, "directory")? {
        client
            .grants
            .push(Grant::directory(new_id("rule")?, value, role));
    }
    for value in option_texts(args, "file")? {
        client
            .grants
            .push(Grant::file(new_id("rule")?, value, role));
    }
    Ok(())
}

fn append_denies(args: &Arguments, client: &mut ClientPolicy) -> Result<(), String> {
    for value in option_texts(args, "deny-directory")? {
        client.denies.push(Deny::directory(new_id("rule")?, value));
    }
    for value in option_texts(args, "deny-file")? {
        client.denies.push(Deny::file(new_id("rule")?, value));
    }
    Ok(())
}

fn option_texts<'a>(args: &'a Arguments, key: &str) -> Result<Vec<&'a str>, String> {
    args.values
        .get(key)
        .into_iter()
        .flatten()
        .map(|value| {
            value
                .to_str()
                .ok_or_else(|| format!("--{key} must be valid UTF-8"))
        })
        .collect()
}

fn credential_metadata(credential: &Credential) -> Value {
    json!({"id":credential.id,"created_at":credential.created_at,
        "expires_at":credential.expires_at,"revoked":credential.revoked})
}

fn credential_status(credential: &Credential) -> &'static str {
    if credential.revoked {
        "revoked"
    } else if credential
        .expires_at
        .is_some_and(|expiry| expiry <= chrono::Utc::now().timestamp())
    {
        "expired"
    } else {
        "active"
    }
}

fn client_metadata(id: &str, client: &ClientPolicy) -> Value {
    json!({"id":id,"enabled":client.enabled,"grants":client.grants,"denies":client.denies,
        "credentials":client.credentials.iter().map(credential_metadata).collect::<Vec<_>>()})
}

fn parse_transport(value: &str) -> Result<&'static str, String> {
    match value {
        "http" => Ok("http"),
        "stdio" => Ok("stdio"),
        _ => Err("transport must be http or stdio".into()),
    }
}

fn parse_role(value: &str) -> Result<Role, String> {
    match value {
        "reader" => Ok(Role::Reader),
        "editor" => Ok(Role::Editor),
        _ => Err("role must be reader or editor".into()),
    }
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::Reader => "reader",
        Role::Editor => "editor",
    }
}

fn parse_action(value: &str) -> Result<AccessAction, String> {
    match value {
        "read" => Ok(AccessAction::Read),
        "create" => Ok(AccessAction::Create),
        "update" => Ok(AccessAction::Update),
        "delete" => Ok(AccessAction::Delete),
        _ => Err("action must be read, create, update or delete".into()),
    }
}

fn new_id(prefix: &str) -> Result<String, String> {
    let mut bytes = [0u8; 12];
    getrandom::fill(&mut bytes).map_err(|_| "could not generate a safe identifier")?;
    Ok(format!(
        "{prefix}_{}",
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn default_token_path(profile: &Path, name: &str) -> PathBuf {
    let suffix = chrono::Utc::now().timestamp_micros();
    profile
        .parent()
        .unwrap_or(Path::new("/"))
        .join("tokens")
        .join(format!("{name}-{suffix}.token"))
}

fn selector(directory: &Option<String>, file: &Option<String>) -> String {
    if let Some(directory) = directory {
        if directory.is_empty() {
            "whole vault".into()
        } else {
            format!("directory {directory}")
        }
    } else {
        format!("file {}", file.as_deref().unwrap_or("invalid"))
    }
}

fn quote_posix(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn quote(value: &str) -> String {
    #[cfg(windows)]
    {
        format!("'{}'", value.replace('\'', "''"))
    }
    #[cfg(not(windows))]
    {
        quote_posix(value)
    }
}

fn quote_path(path: &Path) -> String {
    quote(&path.to_string_lossy())
}

fn validated_url(value: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(value)
        .map_err(|_| "--url must be an absolute HTTP or HTTPS MCP URL")?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("--url must be HTTP or HTTPS, without credentials, query or fragment".into());
    }
    if url.port() == Some(0) {
        return Err("--url must use a fixed nonzero port; omit it during setup to select a free default port".into());
    }
    if url.scheme() == "http"
        && !matches!(url.host_str(), Some("127.0.0.1" | "[::1]" | "localhost"))
    {
        return Err("Plain HTTP must use loopback. Use HTTPS for a network endpoint or a protected SSH forward to loopback.".into());
    }
    Ok(url)
}

fn executable_invocation(binary: &Path) -> String {
    #[cfg(windows)]
    {
        format!("& {}", quote_path(binary))
    }
    #[cfg(not(windows))]
    {
        quote_path(binary)
    }
}

fn launch_command(profile: &Path, vault: &Path, url: &str) -> Result<String, String> {
    let url = validated_url(url)?;
    if url.scheme() != "http" || url.path() != "/mcp" {
        return Err("server launch commands require a loopback HTTP /mcp backend".into());
    }
    let port = url
        .port_or_known_default()
        .ok_or("backend endpoint has no port")?;
    let binary = std::env::current_exe().map_err(|error| error.to_string())?;
    let host = local_setup_host(&url);
    Ok(format!(
        "{} --http --host {host} --port {port} --access-file {} {}",
        executable_invocation(&binary),
        quote_path(profile),
        quote_path(vault)
    ))
}

fn doctor_command(profile: &Path, token: &Path, url: &reqwest::Url) -> Result<String, String> {
    let binary = std::env::current_exe().map_err(|error| error.to_string())?;
    Ok(format!(
        "{} access doctor --access-file {} --token-file {} --url {}",
        executable_invocation(&binary),
        quote_path(profile),
        quote_path(token),
        quote(url.as_str())
    ))
}

fn prompt(label: &str, default: Option<&str>) -> Result<String, String> {
    if let Some(default) = default {
        eprint!("{label} [{default}]: ");
    } else {
        eprint!("{label}: ");
    }
    io::stderr().flush().map_err(|error| error.to_string())?;
    let mut line = String::new();
    if io::stdin()
        .read_line(&mut line)
        .map_err(|error| error.to_string())?
        == 0
    {
        return Err("Setup input ended; no setup was confirmed".into());
    }
    let line = line.trim();
    Ok(if line.is_empty() {
        default.unwrap_or("").to_owned()
    } else {
        line.to_owned()
    })
}

fn prompt_choice(label: &str, default: &str, choices: &[&str]) -> Result<String, String> {
    let label = format!("{label}: {}", choices.join(" or "));
    loop {
        let choice = prompt(&label, Some(default))?;
        if choices.contains(&choice.as_str()) {
            return Ok(choice);
        }
        eprintln!("Choose {}.", choices.join(" or "));
    }
}

fn print_help() {
    println!(
        "obsidian-mcp access — optional authenticated clients

USAGE:
  obsidian-mcp access setup VAULT
  obsidian-mcp access init VAULT
  obsidian-mcp access COMMAND --access-file PATH

COMMANDS:
  setup VAULT                 Guided client, token and configuration setup
  init VAULT                  Create a protected external access profile
  client add NAME             Create a client and deliver its first token
  client list                 List clients, rule IDs and credential IDs
  client enable|disable NAME  Change client availability
  client remove NAME          Remove a client and all its credentials
  grant NAME                  Add --directory or --file and --role reader|editor
  deny NAME                   Add --directory or --file denial
  rules remove NAME RULE_ID   Remove one grant or denial
  token create NAME           Deliver another token; existing tokens stay valid
  token list NAME             List safe credential metadata
  token revoke NAME ID        Revoke exactly one credential
  check NAME                  Explain --action read|create|update|delete --path PATH
  preview NAME                List permitted paths without reading note contents
  config NAME                 Write a client configuration snippet
  doctor                      Check unauthenticated refusal and authenticated MCP
  validate                    Validate profile protection, policy and vault binding
  rebind --vault VAULT        Rebind after an intentional move/restore; restart server
  headers --token-file PATH   Emit header JSON for a configured client helper

OPTIONS:
  --access-file PATH          Explicit profile (or OBSIDIAN_ACCESS_FILE)
  --directory PATH            Recursive grant; empty string means whole vault
  --file PATH                 Exact-file grant; repeat resource options as needed
  --role reader|editor        Reader by default when adding a client
  --deny-directory PATH       Deny a subtree during client creation/setup
  --deny-file PATH            Deny an exact file during client creation/setup
  --token-out PATH            Protected token delivery file for client add/setup
  --token-file PATH           Existing protected token file for config/doctor/helper
  --out PATH                  New token or configuration file; never overwrites
  --transport http|stdio      HTTP by default; stdio uses local policy and source
  --client codex|claude-code  Native client configuration format
  --client-token-file PATH    Token location on the client machine
  --binary PATH               Binary location on the client machine for helper
  --url URL                   MCP endpoint (default http://127.0.0.1:37842/mcp)
  --expires-at RFC3339        Optional future credential expiration
  --expected-revision NUMBER  Reject changes based on an outdated review
  --dry-run                   Validate a change without writing files
  --json                      Machine-readable safe output
  --semantic                  Doctor explicitly runs a semantic query
  --write-check PATH          Doctor explicitly creates/reads/deletes a disposable note

Authentication requires an explicit --access-file or OBSIDIAN_ACCESS_FILE setting.
Creating a profile does not change existing servers or client configuration.
MCP-local semantic ownership may still use your configured external API provider.
Header helpers require support in the installed client; OAuth-only clients are not supported."
    );
}
