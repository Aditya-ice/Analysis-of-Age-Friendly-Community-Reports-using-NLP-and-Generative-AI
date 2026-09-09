-- Run as the migration owner after Alembic, in a dedicated ElderHelp project.
-- No passwords here. Create separate LOGIN users privately and grant these groups.
DO $roles$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'elderhelp_serving') THEN
    CREATE ROLE elderhelp_serving NOLOGIN NOSUPERUSER NOBYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'elderhelp_ingestion') THEN
    CREATE ROLE elderhelp_ingestion NOLOGIN NOSUPERUSER NOBYPASSRLS;
  END IF;
END
$roles$;
GRANT USAGE ON SCHEMA public TO elderhelp_serving, elderhelp_ingestion;
DO $permissions$
DECLARE
  table_name text;
  public_role text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY[
    'reports', 'pages', 'chunks', 'ingestion_runs', 'report_revisions',
    'index_generations', 'active_corpus', 'research_pages', 'sections',
    'source_spans', 'chunk_embeddings', 'research_chunks', 'generation_chunks',
    'ingestion_jobs', 'demo_invites', 'quota_counters', 'alembic_version'
  ] LOOP
    EXECUTE format('REVOKE ALL ON TABLE public.%I FROM PUBLIC', table_name);
    FOREACH public_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
      IF EXISTS (SELECT FROM pg_roles WHERE rolname = public_role) THEN
        EXECUTE format('REVOKE ALL ON TABLE public.%I FROM %I', table_name, public_role);
      END IF;
    END LOOP;
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
    EXECUTE format('DROP POLICY IF EXISTS elderhelp_read ON public.%I', table_name);
    EXECUTE format('DROP POLICY IF EXISTS elderhelp_admin ON public.%I', table_name);
    EXECUTE format('DROP POLICY IF EXISTS elderhelp_quota ON public.%I', table_name);
    EXECUTE format('GRANT SELECT ON TABLE public.%I TO elderhelp_serving', table_name);
    EXECUTE format('CREATE POLICY elderhelp_read ON public.%I FOR SELECT TO elderhelp_serving USING (true)', table_name);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.%I TO elderhelp_ingestion', table_name);
    EXECUTE format('CREATE POLICY elderhelp_admin ON public.%I FOR ALL TO elderhelp_ingestion USING (true) WITH CHECK (true)', table_name);
  END LOOP;
END
$permissions$;
GRANT INSERT, UPDATE, DELETE ON public.quota_counters TO elderhelp_serving;
CREATE POLICY elderhelp_quota ON public.quota_counters FOR ALL TO elderhelp_serving
  USING (true) WITH CHECK (true);
-- Apply again after adding tables; new tables are not implicitly exposed.
-- The application enforces report approval; server role SELECT is not a public API.
